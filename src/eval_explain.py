"""
Muhafiz.sol - explanation quality (E3b), latency and error analysis (E5)
(blueprint 8.7, P3.5).

    python -m src.eval_explain

E3b hit definitions (per SmartBugs contract whose category maps to a class):
  strict@k  - an annotated vulnerable line is among the top-k highlighted lines
  relaxed@k - an annotated line lies in the same function as a top-k line
A random-ranking baseline (expected hit@k when k code lines are picked at
random) is reported next to each value so the numbers can be interpreted.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import load_config, path, read_json, seed, write_json
from src.evaluate import bootstrap_ci
from src.explain import explain_all, find_blocks
from src.predict import Predictor
from src.preprocess import split_lines
from src.run_experiments import _csv, _log, check_frozen, gold_targets


def _random_hit_rate(n_code_lines: int, n_vuln: int, k: int) -> float:
    """P(at least one of n_vuln lines among k random picks of n_code_lines)."""
    if n_code_lines <= 0:
        return 0.0
    k = min(k, n_code_lines)
    miss = 1.0
    for i in range(k):
        miss *= max(n_code_lines - n_vuln - i, 0) / (n_code_lines - i)
    return 1.0 - miss


def e3b(pred: Predictor) -> pd.DataFrame:
    gold = gold_targets(pred.classes)
    gold = gold[gold["target_class"].notna()].reset_index(drop=True)
    from src.gold_overlap import load as load_overlap

    overlap = load_overlap().set_index("id")["overlap"]
    iters = load_config()["training"]["bootstrap_iterations"]
    per = []
    t0 = time.time()
    for row in gold.itertuples():
        # the model only ever sees the cleaned text; line numbers are identical to the raw file
        src_clean = row.source_clean
        ex = explain_all(src_clean, pred, top_k=3)
        items = ex["lines"][row.target_class]
        top = [it["line"] for it in items]
        vuln = set(int(x) for x in row.vulnerable_lines)
        blocks = find_blocks(src_clean)

        def block_of(ln):
            for b in blocks:
                if b.start_line <= ln <= b.end_line:
                    return (b.start_line, b.end_line)
            return None

        vuln_blocks = {block_of(v) for v in vuln} - {None}
        code_set = [i + 1 for i, toks in enumerate(pred.line_tokens(src_clean)[0]) if toks]
        code_lines = len(code_set)
        # chance that a randomly picked code line lies in a function containing an annotated line (or is one)
        in_vuln_fn = sum(1 for ln in code_set if ln in vuln or block_of(ln) in vuln_blocks)
        f_rel = in_vuln_fn / code_lines if code_lines else 0.0
        per.append({
            "id": row.id, "category": row.category, "class": row.target_class, "status": row.mapping_status,
            "flagged": bool(pred.predict(src_clean)["flags"][row.target_class]),
            "probability": ex["base"].get(row.target_class, np.nan),
            "top_lines": top, "vulnerable_lines": sorted(vuln),
            "strict@1": bool(set(top[:1]) & vuln), "strict@3": bool(set(top[:3]) & vuln),
            "relaxed@1": bool({block_of(t) for t in top[:1]} & vuln_blocks) or bool(set(top[:1]) & vuln),
            "relaxed@3": bool({block_of(t) for t in top[:3]} & vuln_blocks) or bool(set(top[:3]) & vuln),
            "random@1": _random_hit_rate(code_lines, len(vuln), 1),
            "random@3": _random_hit_rate(code_lines, len(vuln), 3),
            "random_relaxed@1": f_rel, "random_relaxed@3": 1 - (1 - f_rel) ** 3,
            "code_lines": code_lines, "overlap_with_training_data": bool(overlap.get(row.id, False)),
        })
    per = pd.DataFrame(per)
    _log(f"E3b explained {len(per)} contracts in {time.time() - t0:.0f}s")
    per.to_json(path("tables_dir") / "e3b_per_contract.json", orient="records", indent=1)

    rows = []
    conf = per[per.status == "confirmed"]
    groups = [("ALL evaluated (confirmed + partial)", per),
              ("confirmed classes", conf),
              ("confirmed, no overlap with training data", conf[~conf["overlap_with_training_data"]]),
              ("confirmed, flagged by the model", conf[conf["flagged"]])] +         [(f"{c} ({per[per.category == c]['status'].iloc[0]})", g) for c, g in per.groupby("category")]
    for name, g in groups:
        r = {"category": name, "contracts": len(g)}
        for key in ("strict@1", "strict@3", "relaxed@1", "relaxed@3"):
            v = g[key].to_numpy(float)
            lo, hi = bootstrap_ci(lambda i: v[i].mean(), len(v), iters, seed=seed())
            r[key] = v.mean()
            r[f"{key}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        for key in ("random@1", "random@3", "random_relaxed@1", "random_relaxed@3"):
            r[key] = g[key].mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    _csv(out, "e3b_hit_rate.csv")
    return per


def e5(pred: Predictor, per: pd.DataFrame) -> None:
    # ---- latency: cold start, warm predict, explanation
    t0 = time.perf_counter()
    cold = Predictor()
    load_s = time.perf_counter() - t0
    samples = sorted((path("models_dir").parent / "app" / "sample_contracts").glob("*.sol"))
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet", columns=["split", "n_tokens"])
    rows = []
    for f in samples:
        src = f.read_text(encoding="utf-8")
        t = time.perf_counter()
        cold.predict(src)
        first = (time.perf_counter() - t) * 1000
        warm = []
        for _ in range(5):
            t = time.perf_counter()
            cold.predict(src)
            warm.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        explain_all(src, cold)
        ex_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        cold.similar(src)
        sim_ms = (time.perf_counter() - t) * 1000
        rows.append({"input": f.name, "lines": src.count("\n") + 1, "first_predict_ms": first,
                     "warm_predict_ms_median": float(np.median(warm)), "explain_ms": ex_ms, "similar_ms": sim_ms})
    # a long real contract: SmartBugs' longest and a training-length synthetic (repeat)
    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet")
    longest = gold.sort_values("n_lines").iloc[-1]
    big = longest["source_clean"]
    for name, src in [(f"smartbugs longest ({longest['id']})", big),
                      ("synthetic 10x long file", "\n".join([big] * 10))]:
        t = time.perf_counter()
        cold.predict(src)
        pm = (time.perf_counter() - t) * 1000
        warm = []
        for _ in range(3):
            t = time.perf_counter()
            cold.predict(src)
            warm.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        explain_all(src, cold)
        rows.append({"input": name, "lines": src.count("\n") + 1, "first_predict_ms": pm,
                     "warm_predict_ms_median": float(np.median(warm)),
                     "explain_ms": (time.perf_counter() - t) * 1000, "similar_ms": np.nan})
    lat = pd.DataFrame(rows)
    lat.loc[len(lat)] = {"input": "cold start: load all model files", "lines": np.nan,
                         "first_predict_ms": load_s * 1000, "warm_predict_ms_median": np.nan,
                         "explain_ms": np.nan, "similar_ms": np.nan}
    _csv(lat, "e5_latency.csv")

    # ---- error analysis on the test split by length and compiler generation
    from src.build_data import solidity_minor
    from src.evaluate import binarize

    meta = read_json(path("models_dir") / "thresholds.json")
    primary = meta["primary"]
    ds = pd.read_parquet(path("processed_dir") / "dataset.parquet",
                         columns=["split", "n_tokens", "pragma"] + [f"y_{c}" for c in meta["classes"]])
    test = ds[ds.split == "test"].reset_index(drop=True)
    from src.run_experiments import Frozen, load_test

    fz = Frozen()
    X, Y = load_test()
    fl = binarize(fz.proba(primary, X), fz.thresholds(primary))
    test["len_bucket"] = pd.qcut(test["n_tokens"], 4, labels=["shortest 25%", "short-mid", "mid-long", "longest 25%"])
    test["solidity"] = test["pragma"].map(solidity_minor)
    err = []
    for col in ("len_bucket", "solidity"):
        for val, g in test.groupby(col, observed=True):
            idx = g.index.to_numpy()
            pos = Y[idx].sum()
            tp = (fl[idx] & Y[idx].astype(bool)).sum()
            fp = (fl[idx] & ~Y[idx].astype(bool)).sum()
            err.append({"slice": col, "value": str(val), "contracts": len(idx),
                        "label_positives": int(pos), "positives_per_contract": pos / max(len(idx), 1),
                        "micro_recall": tp / max(pos, 1),
                        "micro_precision": tp / max(tp + fp, 1)})
    _csv(pd.DataFrame(err), "e5_error_slices.csv")

    # ---- case studies: 5 successes, 5 failures on SmartBugs (confirmed classes)
    # only contracts with no copy in the training data, and varied: at most two per class and no two
    # successes with the same highlighted lines (near-identical template contracts)
    conf = per[(per.status == "confirmed") & ~per["overlap_with_training_data"]]

    def varied(df, n):
        out, seen, per_class = [], set(), {}
        for i, r in df.iterrows():
            key = (r["class"], tuple(r["top_lines"]))
            if key in seen or per_class.get(r["class"], 0) >= 2:
                continue
            seen.add(key)
            per_class[r["class"]] = per_class.get(r["class"], 0) + 1
            out.append(i)
            if len(out) == n:
                break
        return df.loc[out]

    succ = varied(conf[conf["flagged"] & conf["strict@3"]].sort_values("probability", ascending=False), 5)
    fail = pd.concat([varied(conf[~conf["flagged"]].sort_values("probability"), 3),
                      varied(conf[conf["flagged"] & ~conf["relaxed@3"]], 2)])
    if len(fail) < 5:
        rest = conf[~conf["strict@3"] & ~conf.index.isin(fail.index)]
        fail = pd.concat([fail, varied(rest, 5 - len(fail))])
    cases = pd.concat([succ.assign(kind="success"), fail.assign(kind="failure")])
    gold_idx = gold.set_index("id")
    records = []
    for r in cases.to_dict("records"):
        lines = split_lines(gold_idx.loc[r["id"], "source_clean"])
        records.append({"kind": r["kind"], "id": r["id"], "class": r["class"], "probability": r["probability"],
                        "flagged": r["flagged"], "top_lines": r["top_lines"], "code_lines": r["code_lines"],
                        "vulnerable_lines": r["vulnerable_lines"],
                        "top_line_text": [lines[i - 1].strip() for i in r["top_lines"]],
                        "vulnerable_text": [lines[i - 1].strip() for i in r["vulnerable_lines"]][:3]})
    write_json(records, path("tables_dir") / "e5_case_studies.json")
    _log("E5 done")


if __name__ == "__main__":
    check_frozen()
    pred = Predictor()
    per = e3b(pred)
    e5(pred, per)
