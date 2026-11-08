"""
Muhafiz.sol - experiments E1-E4 (blueprint 8.7, P2.5).

    python -m src.run_experiments            # E1, E2, E3, E4 (+ figures)
    python -m src.run_experiments --only e1 e3

Guard (acceptance test B2): the frozen model must be tagged ``model-v1`` in
git and models/thresholds.json must be unchanged since the tag, otherwise the
script refuses to read the test split or SmartBugs.

Outputs in reports/tables (long schema model,split,class,metric,value where it
applies) and reports/figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import plotstyle
from src.config import ROOT, load_config, path, read_json, seed, write_json
from src.evaluate import (binarize, calibration_table, choose_thresholds, evaluate, metric_ci, pr_auc,
                          recall_with_ci, summary_metrics)
from src.train import apply_platt

MODEL_ORDER = ["prior", "logreg", "tree", "forest", "knn", "mlp"]
MODEL_NAMES = {"prior": "Majority / prior baseline", "logreg": "Logistic regression", "tree": "Decision tree",
               "forest": "Random forest", "knn": "K-nearest neighbours", "mlp": "MLP neural network"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _csv(df: pd.DataFrame, name: str) -> None:
    out = path("tables_dir") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n", float_format="%.4f")


def check_frozen(allow_untagged: bool = False) -> str:
    """Refuse to evaluate on test/gold unless model-v1 exists and thresholds are unchanged."""
    try:
        tag = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", "refs/tags/model-v1"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "model-v1", "--",
                                "models/thresholds.json"]).returncode if tag else 1
    except FileNotFoundError:
        tag, dirty = "", 1
    if (not tag or dirty) and not allow_untagged:
        raise SystemExit("Refusing to evaluate on test/SmartBugs: tag model-v1 missing or thresholds.json "
                         "changed since the tag (acceptance test B2). Use --allow-untagged for development.")
    # the evaluated copies of the app models must be the files frozen under the tag
    for fam in ("mlp", "logreg"):
        exp, app = path("models_dir") / "experiments" / f"{fam}.joblib", path("models_dir") / f"model_{fam}.joblib"
        if tag and exp.exists() and app.exists() and _sha256(exp) != _sha256(app) and not allow_untagged:
            raise SystemExit(f"models/experiments/{fam}.joblib differs from the frozen models/model_{fam}.joblib")
    return tag or "untagged"


def _sha256(p) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clone_groups_of_test() -> np.ndarray:
    """Clone-group id of every test contract, in the row order of X_test / Y_test."""
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet", columns=["split", "group"])
    return df.loc[df["split"] == "test", "group"].to_numpy()


# --------------------------------------------------------------------------- #
class Frozen:
    """The frozen model family objects + thresholds/calibration from thresholds.json."""

    def __init__(self):
        self.meta = read_json(path("models_dir") / "thresholds.json")
        self.classes = self.meta["classes"]
        self.vec = joblib.load(path("models_dir") / "vectorizer.joblib")
        exp = path("models_dir") / "experiments"
        self.models = {f: joblib.load(exp / f"{f}.joblib") for f in self.meta["all_models"]}

    def uses_svd(self, fam: str) -> bool:
        return fam == "knn" or (fam == "mlp" and self.meta["all_models"]["mlp"]["params"].get("input") == "svd")

    def proba(self, fam: str, X, model=None, platt="frozen") -> np.ndarray:
        m = model or self.models[fam]
        Xin = self.vec.transform_svd(X) if self.uses_svd(fam) else X
        p = np.asarray(m.predict_proba(Xin))
        params = self.meta["all_models"][fam]["platt"] if platt == "frozen" else platt
        return apply_platt(p, params)

    def thresholds(self, fam: str) -> np.ndarray:
        return np.asarray(self.meta["all_models"][fam]["thresholds"])


def load_test(name: str = "test"):
    feat = path("features_dir")
    return sp.load_npz(feat / f"X_{name}.npz").tocsr(), np.load(feat / f"Y_{name}.npy")


def gold_targets(classes: List[str]) -> pd.DataFrame:
    """SmartBugs contracts with the mapped target class and mapping status."""
    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet")
    lm = read_json(path("models_dir") / "label_map.json")["smartbugs_category_mapping"]
    gold["target_class"] = gold["category"].map(lambda c: lm[c]["class"])
    gold["mapping_status"] = gold["category"].map(lambda c: lm[c]["status"])
    return gold


# --------------------------------------------------------------------------- #
def e1(fz: Frozen) -> Dict[str, np.ndarray]:
    X, Y = load_test()
    iters = load_config()["training"]["bootstrap_iterations"]
    groups = clone_groups_of_test()
    assert len(groups) == len(Y)
    long, summary, probs = [], [], {}
    for fam in [f for f in MODEL_ORDER if f in fz.models]:
        p = fz.proba(fam, X)
        probs[fam] = p
        thr = fz.thresholds(fam)
        lg, pc, sm = evaluate(fam, "test", Y, p, thr, fz.classes)
        long.append(lg)
        # cluster bootstrap over clone groups, config iterations (blueprint: about 1,000)
        lo, hi = metric_ci(Y, p, thr, "macro_pr_auc", iterations=iters, seed=seed(), groups=groups)
        flo, fhi = metric_ci(Y, p, thr, "macro_f1", iterations=iters, seed=seed(), groups=groups)
        summary.append({"model": MODEL_NAMES[fam], "key": fam, "macro_pr_auc": sm["macro_pr_auc"],
                        "macro_pr_auc_ci": f"[{lo:.3f}, {hi:.3f}]", "macro_f1": sm["macro_f1"],
                        "macro_f1_ci": f"[{flo:.3f}, {fhi:.3f}]", "macro_f2": sm["macro_f2"],
                        "macro_recall": sm["macro_recall"], "macro_precision": sm["macro_precision"],
                        "micro_f1": sm["micro_f1"], "hamming_loss": sm["hamming_loss"],
                        "label_accuracy": sm["label_accuracy"], "subset_accuracy": sm["subset_accuracy"]})
        pc.insert(0, "model", fam)
        _csv(pc, f"e1_per_class_{fam}.csv")
        _log(f"E1 {fam}: macro PR-AUC {sm['macro_pr_auc']:.4f} macro F1 {sm['macro_f1']:.4f}")
    # MLP seed spread on test (B5)
    seed_rows = []
    for f in sorted((path("models_dir") / "experiments").glob("mlp_seed*.joblib")):
        b = joblib.load(f)
        Xin = fz.vec.transform_svd(X) if b["input"] == "svd" else X
        p = apply_platt(np.asarray(b["model"].predict_proba(Xin)), b["platt"])
        sm = summary_metrics(Y, p, b["thresholds"])
        seed_rows.append({"seed": b["seed"], "input": b["input"], "test_macro_pr_auc": sm["macro_pr_auc"],
                          "test_macro_f1": sm["macro_f1"], "test_macro_f2": sm["macro_f2"]})
    if seed_rows:
        sd = pd.DataFrame(seed_rows).astype({"seed": object})
        sd.loc[len(sd)] = {"seed": "mean", "input": "", **sd.drop(columns=["seed", "input"]).mean().to_dict()}
        sd.loc[len(sd)] = {"seed": "std", "input": "",
                           **sd.iloc[:-1].drop(columns=["seed", "input"]).astype(float).std(ddof=0).to_dict()}
        _csv(sd, "e1_mlp_seeds_test.csv")
    _csv(pd.concat(long), "e1_test_results.csv")
    _csv(pd.DataFrame(summary), "e1_test_summary.csv")
    primary = fz.meta["primary"]
    _csv(calibration_table(Y, probs[primary], fz.classes), "e1_calibration_primary.csv")
    _plot_e1(pd.DataFrame(summary), probs, Y, fz)
    return probs


def _plot_e1(summary: pd.DataFrame, probs, Y, fz: Frozen) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fams = [f for f in MODEL_ORDER if f in probs]
    aps = np.array([pr_auc(Y, probs[f]) for f in fams])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(fz.classes))
    w = 0.8 / len(fams)
    for k, f in enumerate(fams):
        ax.bar(x + (k - (len(fams) - 1) / 2) * w, aps[k], w * 0.92, color=plotstyle.SERIES[k], label=MODEL_NAMES[f])
    ax.set_xticks(x, fz.classes, rotation=15, ha="right")
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1)
    ax.set_title("E1 - per-class PR-AUC on the cluster-aware test split")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    plotstyle.save(fig, path("figures_dir") / "e1_per_class_pr_auc.png")

    from sklearn.metrics import precision_recall_curve

    primary = fz.meta["primary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, name in enumerate(fz.classes):
        pr, rc, _ = precision_recall_curve(Y[:, c], probs[primary][:, c])
        axes[0].plot(rc, pr, color=plotstyle.SERIES[c % 8], label=name, linewidth=1.6)
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title(f"Precision-recall curves ({MODEL_NAMES[primary]})")
    axes[0].legend(fontsize=8)
    cal = calibration_table(Y, probs[primary], fz.classes)
    for c, name in enumerate(fz.classes):
        d = cal[cal["class"] == name]
        axes[1].plot(d["mean_predicted"], d["observed"], marker="o", markersize=4,
                     color=plotstyle.SERIES[c % 8], label=name, linewidth=1.4)
    axes[1].plot([0, 1], [0, 1], color=plotstyle.INK2, linestyle="--", linewidth=1)
    axes[1].set_xlabel("mean predicted probability")
    axes[1].set_ylabel("observed frequency")
    axes[1].set_title("Calibration (reliability curve)")
    plotstyle.save(fig, path("figures_dir") / "e1_pr_and_calibration.png")


# --------------------------------------------------------------------------- #
def e2(fz: Frozen) -> None:
    """Same configurations retrained on the random split; gap vs cluster split."""
    from src.train import train_all

    fixed = {f: fz.meta["all_models"][f]["params"] for f in fz.meta["all_models"]}
    res = train_all(prefix="random_", fixed=fixed, out_dir=path("models_dir") / "experiments" / "random")
    cfg = load_config()["training"]["threshold"]
    Yva = res["_Yva"]
    Xte, Yte = load_test("random_test")
    Xc, Yc = load_test("test")
    e1_sum = pd.read_csv(path("tables_dir") / "e1_test_summary.csv").set_index("key")
    rows, long = [], []
    for fam in [f for f in MODEL_ORDER if f in res]:
        r = res[fam]
        thr, _ = choose_thresholds(Yva, r["val_proba"], fz.classes, cfg["beta"], cfg["min_precision"])
        Xin = fz.vec.transform_svd(Xte) if fz.uses_svd(fam) else Xte
        p = np.asarray(r["model"].predict_proba(Xin))
        lg, _, sm = evaluate(fam, "random_test", Yte, p, thr, fz.classes)
        long.append(lg)
        rows.append({"model": MODEL_NAMES[fam], "key": fam,
                     "cluster_test_macro_pr_auc": e1_sum.loc[fam, "macro_pr_auc"],
                     "random_test_macro_pr_auc": sm["macro_pr_auc"],
                     "pr_auc_inflation": sm["macro_pr_auc"] - e1_sum.loc[fam, "macro_pr_auc"],
                     "cluster_test_macro_f1": e1_sum.loc[fam, "macro_f1"], "random_test_macro_f1": sm["macro_f1"],
                     "f1_inflation": sm["macro_f1"] - e1_sum.loc[fam, "macro_f1"]})
        _log(f"E2 {fam}: random {sm['macro_pr_auc']:.4f} vs cluster {e1_sum.loc[fam, 'macro_pr_auc']:.4f}")
    summary = read_json(path("processed_dir") / "build_summary.json")["dedup"]
    out = pd.DataFrame(rows)
    _csv(out, "e2_leakage.csv")
    _csv(pd.concat(long), "e2_random_test_results.csv")
    write_json({"random_split_leakage_rate": summary["random_split_leakage_rate"],
                "group_split_leakage_rate": summary["group_split_leakage_rate"]},
               path("tables_dir") / "e2_leakage_rates.json")

    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(out))
    ax.bar(x - 0.2, out["cluster_test_macro_pr_auc"], 0.38, color=plotstyle.SERIES[0], label="cluster-aware split")
    ax.bar(x + 0.2, out["random_test_macro_pr_auc"], 0.38, color=plotstyle.SERIES[1], label="random split")
    for i, v in enumerate(out["pr_auc_inflation"]):
        ax.text(i + 0.2, out["random_test_macro_pr_auc"].iloc[i] + 0.015, f"{v:+.3f}", ha="center", fontsize=8,
                color=plotstyle.INK2)
    ax.set_xticks(x, out["model"], rotation=15, ha="right")
    ax.set_ylabel("macro PR-AUC (test)")
    ax.set_ylim(0, 1)
    ax.set_title("E2 - clone leakage: random split vs cluster-aware split")
    ax.legend()
    plotstyle.save(fig, path("figures_dir") / "e2_leakage.png")


# --------------------------------------------------------------------------- #
def gold_docs(gold: pd.DataFrame, keep_comments: bool = False, raw: bool = False) -> List[str]:
    from src.preprocess import normalise, tokenize, truncate_tokens

    pre = load_config()["preprocessing"]
    docs = []
    for row in gold.itertuples():
        text = row.source_clean
        if keep_comments:
            text = (path("smartbugs_dir") / row.path).read_text(encoding="utf-8", errors="replace")
        toks, _ = truncate_tokens(tokenize(normalise(text)), pre["max_tokens"], pre["head_fraction"])
        docs.append(" ".join(toks))
    return docs


def e3(fz: Frozen, test_probs: Dict[str, np.ndarray]) -> None:
    from src.gold_overlap import load as load_overlap

    gold = gold_targets(fz.classes)
    gold["overlap"] = gold["id"].map(load_overlap().set_index("id")["overlap"]).fillna(False).astype(bool)
    Xg = sp.load_npz(path("features_dir") / "X_gold.npz").tocsr()
    iters = load_config()["training"]["bootstrap_iterations"]
    rows, allrows = [], []
    primary = fz.meta["primary"]
    for fam in [f for f in MODEL_ORDER if f in fz.models]:
        p = fz.proba(fam, Xg)
        flags = binarize(p, fz.thresholds(fam))
        for cat, grp in gold.groupby("category"):
            cls, status = grp["target_class"].iloc[0], grp["mapping_status"].iloc[0]
            cls = cls if isinstance(cls, str) and cls else None  # unmapped categories arrive as None or NaN
            row = {"model": fam, "smartbugs_category": cat, "mapped_class": cls or "-", "status": status,
                   "contracts": len(grp)}
            if cls:
                c = fz.classes.index(cls)
                yt = np.ones(len(grp), dtype=int)
                r, lo, hi, n = recall_with_ci(yt, flags[grp.index, c], iters, seed())
                clean = grp[~grp["overlap"]]
                rc, clo, chi, nc = recall_with_ci(np.ones(len(clean), dtype=int), flags[clean.index, c], iters,
                                                  seed())
                row.update({"recall": r, "ci_low": lo, "ci_high": hi,
                            "mean_probability": float(p[grp.index, c].mean()),
                            "any_flag_rate": float(flags[grp.index].any(1).mean()),
                            "contracts_no_overlap": nc, "recall_no_overlap": rc,
                            "ci_low_no_overlap": clo, "ci_high_no_overlap": chi})
            else:
                row.update({"recall": np.nan, "ci_low": np.nan, "ci_high": np.nan, "mean_probability": np.nan,
                            "any_flag_rate": float(flags[grp.index].any(1).mean())})
            allrows.append(row)
        np.save(path("features_dir") / f"gold_proba_{fam}.npy", p)
    allr = pd.DataFrame(allrows)
    _csv(allr, "e3_smartbugs_all_models.csv")
    prim = allr[allr["model"] == primary].drop(columns="model")
    order = {"confirmed": 0, "partial": 1, "not evaluated": 2}
    prim = prim.sort_values(["status", "contracts"], key=lambda s: s.map(order) if s.name == "status" else -s)
    _csv(prim, "e3_smartbugs_recall.csv")

    # pooled recall over confirmed classes per model + false-positive rate on the test split
    X, Y = load_test()
    fp_rows = []
    for fam in [f for f in MODEL_ORDER if f in fz.models]:
        conf = allr[(allr.model == fam) & (allr.status == "confirmed")]
        pooled = float((conf["recall"] * conf["contracts"]).sum() / conf["contracts"].sum())
        pooled_clean = float((conf["recall_no_overlap"] * conf["contracts_no_overlap"]).sum()
                             / conf["contracts_no_overlap"].sum())
        clean = Y.sum(1) == 0
        fl = binarize(test_probs[fam], fz.thresholds(fam))
        fp_rows.append({"model": MODEL_NAMES[fam], "key": fam, "smartbugs_pooled_recall_confirmed": pooled,
                        "smartbugs_pooled_recall_confirmed_no_overlap": pooled_clean,
                        "smartbugs_confirmed_contracts_no_overlap": int(conf["contracts_no_overlap"].sum()),
                        "test_clean_contracts": int(clean.sum()),
                        "false_alarm_rate_on_clean_test": float(fl[clean].any(1).mean()),
                        "macro_fpr_test": float(np.mean([(fl[:, c] & (Y[:, c] == 0)).sum() / max((Y[:, c] == 0).sum(), 1)
                                                         for c in range(len(fz.classes))]))})
    _csv(pd.DataFrame(fp_rows), "e3_summary.csv")
    _log("E3 done")


# --------------------------------------------------------------------------- #
def e4(fz: Frozen) -> None:
    """Ablations with logistic regression (fast to retrain)."""
    from src.build_data import gold_token_docs
    from src.features import FeaturePipeline
    from src.models import LogRegModel

    cfg = load_config()
    fcfg, pre, tcfg = cfg["features"], cfg["preprocessing"], cfg["training"]["threshold"]
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet",
                         columns=["split", "tokens"] + [f"y_{c}" for c in fz.classes])
    gold = gold_targets(fz.classes)
    conf = gold[gold["mapping_status"] == "confirmed"]
    C = fz.meta["all_models"]["logreg"]["params"]["C"]
    rows = []

    def gold_recall(p, thr):
        fl = binarize(p, thr)
        hits = [fl[i, fz.classes.index(c)] for i, c in zip(conf.index, conf["target_class"])]
        return float(np.mean(hits))

    def record(ablation, setting, p_test, Yte, thr, p_gold):
        sm = summary_metrics(Yte, p_test, thr)
        rows.append({"ablation": ablation, "setting": setting, "test_macro_pr_auc": sm["macro_pr_auc"],
                     "test_macro_f1": sm["macro_f1"], "smartbugs_recall_confirmed": gold_recall(p_gold, thr)})
        _log(f"E4 {ablation} / {setting}: {rows[-1]}")

    # reference: the frozen logistic regression (ids off, 1-3 grams, comments stripped)
    Xte, Yte = load_test()
    Xg = sp.load_npz(path("features_dir") / "X_gold.npz").tocsr()
    thr_lr = fz.thresholds("logreg")
    p_te = fz.proba("logreg", Xte)
    p_g = fz.proba("logreg", Xg)
    ref = ("reference", "comments stripped, identifiers kept, n-grams 1-3")
    record(*ref, p_te, Yte, thr_lr, p_g)

    # (a) comments kept on the gold set: annotation markers reach the model input
    Xg_raw = fz.vec.transform(gold_docs(gold, keep_comments=True))
    record("comments", "comments KEPT on SmartBugs (annotations visible)", p_te, Yte, thr_lr,
           fz.proba("logreg", Xg_raw))

    # (b) identifier normalisation and (c) n-gram range: refit vectoriser + LR on train
    rng = np.random.default_rng(seed())
    train = df[df.split == "train"]
    vocab_docs = list(train["tokens"].to_numpy()[rng.choice(len(train), min(fcfg["vocab_fit_docs"], len(train)),
                                                              replace=False)])
    for ablation, setting, kw in [("identifiers", "user identifiers -> <ID>", {"ids": True}),
                                  ("ngrams", "n-grams 1-2", {"ngram_range": (1, 2)}),
                                  ("ngrams", "n-grams 1-1 (bag of tokens)", {"ngram_range": (1, 1)})]:
        params = dict(ngram_range=tuple(fcfg["ngram_range"]), max_features=fcfg["max_features"],
                      min_df=fcfg["min_df"], max_df=fcfg["max_df"], sublinear_tf=fcfg["sublinear_tf"],
                      ids=bool(fcfg.get("normalise_identifiers", False)), random_state=seed())
        params.update(kw)
        vec = FeaturePipeline(**params).fit(train["tokens"].tolist(), vocab_docs=vocab_docs)
        Ys = {s: df[df.split == s][[f"y_{c}" for c in fz.classes]].to_numpy() for s in ("train", "val", "test")}
        Xs = {s: vec.transform(df[df.split == s]["tokens"].tolist()) for s in ("train", "val", "test")}
        m = LogRegModel(C=C).fit(Xs["train"], Ys["train"])
        pv = m.predict_proba(Xs["val"])
        thr, _ = choose_thresholds(Ys["val"], pv, fz.classes, tcfg["beta"], tcfg["min_precision"])
        pg = m.predict_proba(vec.transform(gold_token_docs(gold, pre["max_tokens"], pre["head_fraction"])))
        record(ablation, setting, m.predict_proba(Xs["test"]), Ys["test"], thr, pg)
    _csv(pd.DataFrame(rows), "e4_ablations.csv")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["e1", "e2", "e3", "e4"])
    ap.add_argument("--allow-untagged", action="store_true")
    a = ap.parse_args()
    tag = check_frozen(a.allow_untagged)
    _log(f"evaluating frozen model ({tag})")
    which = a.only or ["e1", "e2", "e3", "e4"]
    fz = Frozen()
    probs = e1(fz) if ("e1" in which or "e3" in which) else {}
    if "e3" in which:
        e3(fz, probs)
    if "e4" in which:
        e4(fz)
    if "e2" in which:
        e2(fz)
