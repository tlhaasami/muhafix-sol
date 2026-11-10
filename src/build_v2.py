"""
model-v2 data pack: Dataset A without any contract whose clone group overlaps SmartBugs Curated.

    python -m src.build_v2          # writes data/processed_v2, models/v2 (label map, vectorizer), data/features_v2

model-v1 was frozen and evaluated before the overlap between SmartBugs and Dataset A was found (review, see
src/gold_overlap.py). model-v2 removes the overlap at the source so the external test is fully independent:
  1. every Dataset A contract that matches a SmartBugs contract (same normalised token hash, or estimated
     Jaccard >= the de-duplication threshold) is found, and its whole clone group is removed;
  2. all other contracts keep their split from model-v1 (no re-splitting, so the comparison stays like-for-like);
  3. the TF-IDF / SVD pipeline is refitted on the reduced training split.
Training then reuses model-v1's chosen hyper-parameters (python -m src.train --fixed-from models/thresholds.json),
so nothing is tuned after seeing test results.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

os.environ["MUHAFIZ_VARIANT"] = "v2"

from src.config import ROOT, load_config, path, read_json, write_json  # noqa: E402


def excluded_groups() -> pd.DataFrame:
    from src.build_data import gold_token_docs
    from src.splits import dedup_tokens, exact_hash, minhash_signature

    cfg = load_config()
    pre, dd = cfg["preprocessing"], cfg["deduplication"]
    tok = pd.read_parquet(path("interim_dir") / "tokens.parquet", columns=["address", "exact_hash", "signature"])
    ds = pd.read_parquet(ROOT / load_config()["paths"]["processed_dir"] / "dataset.parquet",
                         columns=["address", "split", "group"])
    rep = tok.merge(ds, on="address")
    sigs = np.frombuffer(b"".join(rep["signature"].tolist()), dtype=np.uint64).reshape(len(rep), -1)
    hashes = rep["exact_hash"].to_numpy()
    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet", columns=["id", "source_clean"])
    rows = []
    for gid, doc in zip(gold["id"], gold_token_docs(gold, pre["max_tokens"], pre["head_fraction"])):
        d = dedup_tokens(doc.split())
        sig = minhash_signature(d, dd["shingle_size"], dd["minhash_num_perm"])
        sim = (sigs == sig[None, :]).mean(1)
        idx = set(np.flatnonzero(sim >= dd["jaccard_threshold"])) | set(np.flatnonzero(hashes == exact_hash(d)))
        for i in idx:
            rows.append({"smartbugs_id": gid, "address": rep["address"].iloc[i], "group": int(rep["group"].iloc[i]),
                         "split": rep["split"].iloc[i], "jaccard": round(float(sim[i]), 4)})
    return pd.DataFrame(rows, columns=["smartbugs_id", "address", "group", "split", "jaccard"])


def main() -> None:
    v1_processed = ROOT / load_config()["paths"]["processed_dir"]
    v1_models = ROOT / load_config()["paths"]["models_dir"]
    out = path("processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    path("models_dir").mkdir(parents=True, exist_ok=True)
    path("tables_dir").mkdir(parents=True, exist_ok=True)
    path("figures_dir").mkdir(parents=True, exist_ok=True)

    matches = excluded_groups()
    groups = set(matches["group"])
    ds = pd.read_parquet(v1_processed / "dataset.parquet")
    drop = ds["group"].isin(groups)
    excl = ds.loc[drop, ["address", "group", "split", "split_random"]].merge(
        matches.groupby("group")["smartbugs_id"].apply(lambda s: ";".join(sorted(set(s)))).reset_index(), on="group")
    excl.to_csv(out / "smartbugs_exclusion.csv", index=False, lineterminator="\n")
    kept = ds.loc[~drop].reset_index(drop=True)
    kept.to_parquet(out / "dataset.parquet", index=False)

    summary = read_json(v1_processed / "build_summary.json")
    summary["funnel"]["after_removing_smartbugs_overlap"] = int(len(kept))
    summary["split_sizes"] = {s: int((kept["split"] == s).sum()) for s in ("train", "val", "test")}
    summary["smartbugs_exclusion"] = {
        "clone_groups_removed": len(groups), "contracts_removed": int(drop.sum()),
        "removed_by_split": {s: int(((ds["split"] == s) & drop).sum()) for s in ("train", "val", "test")},
        "smartbugs_contracts_matched": int(matches["smartbugs_id"].nunique())}
    summary.pop("split_hashes", None)
    write_json(summary, out / "build_summary.json")
    shutil.copy(v1_models / "label_map.json", path("models_dir") / "label_map.json")
    print(f"removed {int(drop.sum())} contracts in {len(groups)} clone groups "
          f"({summary['smartbugs_exclusion']['removed_by_split']}); kept {len(kept):,}")

    from src.build_data import stage_features

    stage_features(force=True)
    print("v2 features written to", path("features_dir"))


if __name__ == "__main__":
    main()
