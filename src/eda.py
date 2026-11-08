"""
Muhafiz.sol - data audit tables and EDA figures (blueprint P1.2 / P1.8).

    python -m src.eda      (also run as the last stage of src.build_data)

Tables (reports/tables)
  dataset_funnel.csv          rows kept at each filtering step
  label4_audit.csv            what upstream id 4 means in each configuration
  detector_frequency.csv      Slither detector -> contracts, family
  family_distribution.csv     label families in the de-duplicated pool
  label_distribution.csv      positives per class per split (cluster + random)
  length_stats.csv            token / line length percentiles, truncation count
  solidity_versions.csv       compiler generation: training pool vs SmartBugs
Figures (reports/figures)
  label_distribution.png, length_histogram.png, duplicate_groups.png,
  solidity_versions.png, label_cooccurrence.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import plotstyle
from src.config import path, read_json
from src.labels import UPSTREAM_LABELS, upstream_ids


def _csv(df: pd.DataFrame, name: str) -> None:
    out = path("tables_dir") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n", float_format="%.6g")


def label4_audit(raw: pd.DataFrame, upstream_map: dict) -> pd.DataFrame:
    """Re-create upstream integer ids for every successful contract and report
    how id 4 behaves in each configuration."""
    rows = []
    for config, subset in (("big", raw[raw["hf_config"] == "big"]),
                           ("small", raw[raw["hf_config"] == "small"]),
                           ("all", raw)):
        ids = [upstream_ids(list(d), config, upstream_map) for d in subset["detectors"]]
        has4 = np.array([4 in i for i in ids])
        with_other = np.array([4 in i and len(i) > 1 for i in ids])
        rows.append({
            "config": config, "contracts": len(ids), "label_names": len(UPSTREAM_LABELS[config]),
            "id4_name": UPSTREAM_LABELS[config][4], "id4_share": has4.mean(),
            "id4_with_other_ids_share": with_other.mean(),
            "max_id_seen": max((max(i) for i in ids if i), default=-1),
            "safe_id": UPSTREAM_LABELS[config].index("safe"),
        })
    return pd.DataFrame(rows)


def build_reports() -> None:
    plotstyle.apply()
    import matplotlib.pyplot as plt

    figs = path("figures_dir")
    summary = read_json(path("processed_dir") / "build_summary.json")
    lm = read_json(path("models_dir") / "label_map.json")
    classes = lm["target_classes"]
    upstream_map = read_json(path("slither_raw_dir") / "label_mappings.json")

    raw = pd.read_parquet(path("interim_dir") / "slither_contracts.parquet",
                          columns=["address", "success", "detectors", "hf_config"])
    raw = raw[~raw["address"].duplicated()]
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet",
                         columns=["address", "group", "split", "split_random", "detectors", "families",
                                  "n_tokens", "truncated", "n_lines", "pragma", "exact_group_size"]
                         + [f"y_{c}" for c in classes])
    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet")

    # ---- funnel and label-4 audit
    _csv(pd.DataFrame(list(summary["funnel"].items()), columns=["step", "contracts"]), "dataset_funnel.csv")
    ok = raw[raw["success"]]
    _csv(label4_audit(ok, upstream_map), "label4_audit.csv")

    # ---- detectors and families (de-duplicated pool)
    det = df["detectors"].explode().dropna()
    fam_map = lm["detector_to_family"]
    det_tab = det.value_counts().rename_axis("detector").reset_index(name="contracts")
    det_tab["share"] = det_tab["contracts"] / len(df)
    det_tab["family"] = det_tab["detector"].map(fam_map)
    _csv(det_tab, "detector_frequency.csv")
    fam_tab = pd.DataFrame([{"family": f, **v} for f, v in lm["family_stats_dedup_pool"].items()])
    fam_tab["kept_as_class"] = fam_tab["family"].isin(classes)
    _csv(fam_tab.sort_values("positives", ascending=False), "family_distribution.csv")

    # ---- class distribution per split
    rows = []
    for col in ("split", "split_random"):
        for s in ("train", "val", "test"):
            part = df[df[col] == s]
            for c in classes:
                rows.append({"split_type": "cluster" if col == "split" else "random", "split": s,
                             "class": c, "positives": int(part[f"y_{c}"].sum()), "contracts": len(part),
                             "prevalence": float(part[f"y_{c}"].mean())})
            rows.append({"split_type": "cluster" if col == "split" else "random", "split": s, "class": "safe",
                         "positives": int((part[[f'y_{c}' for c in classes]].sum(axis=1) == 0).sum()),
                         "contracts": len(part),
                         "prevalence": float((part[[f'y_{c}' for c in classes]].sum(axis=1) == 0).mean())})
    dist = pd.DataFrame(rows)
    _csv(dist, "label_distribution.csv")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    cl = dist[(dist.split_type == "cluster")]
    names = classes + ["safe"]
    x = np.arange(len(names))
    w = 0.27
    for k, s in enumerate(("train", "val", "test")):
        vals = [cl[(cl.split == s) & (cl["class"] == n)]["prevalence"].iloc[0] * 100 for n in names]
        ax.bar(x + (k - 1) * w, vals, w - 0.02, color=plotstyle.SPLIT_COLORS[s], label=s)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("% of contracts in split")
    ax.set_title("Class prevalence per split (cluster-aware split)")
    ax.legend(ncol=3, loc="upper right")
    plotstyle.save(fig, figs / "label_distribution.png")

    # ---- co-occurrence
    Y = df[[f"y_{c}" for c in classes]].to_numpy().astype(np.int64)  # int8 labels would overflow in Y.T @ Y
    co = (Y.T @ Y) / np.maximum(Y.sum(0)[:, None], 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(co, cmap=plt.matplotlib.colors.ListedColormap(plotstyle.SEQ), vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{co[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if co[i, j] > 0.55 else plotstyle.INK)
    ax.set_title("P(column class | row class)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plotstyle.save(fig, figs / "label_cooccurrence.png")

    # ---- lengths
    from src.preprocess import normalise, tokenize

    gold_tok = np.array([len(tokenize(normalise(t))) for t in gold["source_clean"]])
    q = [0.5, 0.9, 0.95, 0.99, 1.0]
    lt = pd.DataFrame({
        "dataset": ["slither_dedup"] * len(q) + ["smartbugs"] * len(q),
        "quantile": q * 2,
        "tokens": list(np.quantile(df["n_tokens"], q)) + list(np.quantile(gold_tok, q)),
        "lines": list(np.quantile(df["n_lines"], q)) + list(np.quantile(gold["n_lines"], q)),
    })
    lt.loc[len(lt)] = {"dataset": "slither_dedup", "quantile": -1, "tokens": int(df["truncated"].sum()),
                       "lines": float(df["truncated"].mean())}
    _csv(lt, "length_stats.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.logspace(1, np.log10(max(df["n_tokens"].max(), 10) + 1), 50)
    ax.hist(df["n_tokens"], bins=bins, color=plotstyle.SERIES[0], alpha=0.85, label="Slither (training pool)",
            density=True, edgecolor=plotstyle.SURFACE, linewidth=0.5)
    ax.hist(gold_tok, bins=bins, color=plotstyle.SERIES[1], alpha=0.75, label="SmartBugs (gold)",
            density=True, edgecolor=plotstyle.SURFACE, linewidth=0.5)
    from src.config import load_config

    cap = load_config()["preprocessing"]["max_tokens"]
    ax.axvline(cap, color=plotstyle.INK2, linestyle="--", linewidth=1)
    ax.text(cap * 1.05, ax.get_ylim()[1] * 0.85, f"token budget {cap:,}", color=plotstyle.INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("tokens per contract (log scale)")
    ax.set_ylabel("density")
    ax.set_title("Contract length: training contracts are far longer than SmartBugs")
    ax.legend()
    plotstyle.save(fig, figs / "length_histogram.png")

    # ---- duplicate groups
    from src.build_data import solidity_minor

    g = df.groupby("group").size()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    edges = [1, 2, 3, 6, 11, 51, 201, 10**9]
    labels = ["1", "2", "3-5", "6-10", "11-50", "51-200", ">200"]
    counts = pd.cut(g, edges, right=False, labels=labels).value_counts().reindex(labels)
    docs = pd.cut(g, edges, right=False, labels=labels)
    doc_share = g.groupby(docs, observed=False).sum().reindex(labels) / len(df) * 100
    ax.bar(labels, doc_share.values, color=plotstyle.SERIES[0], width=0.7)
    for i, (v, c) in enumerate(zip(doc_share.values, counts.values)):
        ax.text(i, v + 0.5, f"{v:.1f}%\n({c:,} groups)", ha="center", fontsize=8, color=plotstyle.INK2)
    ax.set_xlabel("near-duplicate group size (contracts)")
    ax.set_ylabel("% of contracts")
    ax.set_title("Where contracts sit: singleton vs clone groups (after exact de-dup)")
    ax.set_ylim(0, doc_share.max() * 1.25)
    plotstyle.save(fig, figs / "duplicate_groups.png")

    # ---- solidity versions
    tv = df["pragma"].map(solidity_minor).value_counts(normalize=True)
    gv = gold["pragma"].map(solidity_minor).value_counts(normalize=True)
    vers = pd.DataFrame({"slither_dedup": tv, "smartbugs": gv}).fillna(0)
    vers = vers.loc[sorted(vers.index, key=lambda s: (s == "unknown", s))]
    _csv(vers.reset_index(names="solidity_minor"), "solidity_versions.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xi = np.arange(len(vers))
    ax.bar(xi - 0.2, vers["slither_dedup"] * 100, 0.38, color=plotstyle.SERIES[0], label="Slither (training pool)")
    ax.bar(xi + 0.2, vers["smartbugs"] * 100, 0.38, color=plotstyle.SERIES[1], label="SmartBugs (gold)")
    ax.set_xticks(xi, vers.index)
    ax.set_ylabel("% of contracts")
    ax.set_xlabel("Solidity compiler generation (from pragma)")
    ax.set_title("Domain shift: SmartBugs is almost all Solidity 0.4")
    ax.legend()
    plotstyle.save(fig, figs / "solidity_versions.png")
    print("[eda] tables and figures written")


if __name__ == "__main__":
    build_reports()
