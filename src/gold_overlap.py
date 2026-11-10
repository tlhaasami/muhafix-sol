"""
Overlap between the SmartBugs gold set and Dataset A (review finding, run after the model-v1 freeze).

    python -m src.gold_overlap        # writes reports/tables/gold_overlap.csv

Many SmartBugs files are real Etherscan contracts, so some of them (or near copies) are also in the
Slither dataset the model was trained on. A gold contract counts as overlapping when any of these holds:
  * exact      - same normalised token hash as a Dataset A contract (the de-duplication key)
  * near       - estimated Jaccard similarity of token 5-gram shingles >= the de-duplication threshold
  * address    - its file name is the address of a Dataset A contract
The split of the best-matching Dataset A contract is recorded. E3 and E3b report results on all
contracts and on the contracts without any overlap ("clean"), which is the honest independent check.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import load_config, path

ADDR = re.compile(r"0x[0-9a-fA-F]{40}")


def compute() -> pd.DataFrame:
    from src.build_data import gold_token_docs
    from src.splits import dedup_tokens, exact_hash, minhash_signature

    cfg = load_config()
    pre, dd = cfg["preprocessing"], cfg["deduplication"]
    k, num_perm, thr = dd["shingle_size"], dd["minhash_num_perm"], dd["jaccard_threshold"]

    tok = pd.read_parquet(path("interim_dir") / "tokens.parquet", columns=["address", "exact_hash", "signature"])
    ds = pd.read_parquet(path("processed_dir") / "dataset.parquet", columns=["address", "split"])
    # every Dataset A contract (also removed exact duplicates) -> split of its representative
    hash_split = tok.merge(ds, on="address", how="inner").drop_duplicates("exact_hash").set_index("exact_hash")["split"]
    tok["split"] = tok["exact_hash"].map(hash_split)
    # only contracts that are in the data pack count (filtered-out sources, and for model-v2 the removed
    # SmartBugs-overlap groups, were never trained on)
    tok = tok[tok["split"].notna()].reset_index(drop=True)
    sigs = np.frombuffer(b"".join(tok["signature"].tolist()), dtype=np.uint64).reshape(len(tok), -1)
    addr_split = dict(zip(tok["address"].str.lower(), tok["split"]))

    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet", columns=["id", "category", "source_clean"])
    docs = gold_token_docs(gold, pre["max_tokens"], pre["head_fraction"])
    rows = []
    for gid, cat, doc in zip(gold["id"], gold["category"], docs):
        d = dedup_tokens(doc.split())
        h = exact_hash(d)
        sig = minhash_signature(d, k, num_perm)
        sim = (sigs == sig[None, :]).mean(1)
        best = int(np.argmax(sim))
        m = ADDR.search(gid)
        addr = m.group(0).lower() if m else None
        exact = h in hash_split.index
        near = bool(sim[best] >= thr)
        same_addr = addr in addr_split if addr else False
        split = (hash_split[h] if exact else tok["split"].iloc[best] if near else
                 addr_split.get(addr) if same_addr else None)
        rows.append({"id": gid, "category": cat, "exact_match": exact, "best_jaccard": round(float(sim[best]), 4),
                     "near_duplicate": near, "same_address": same_addr,
                     "overlap": exact or near or same_addr, "matched_split": split})
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    return pd.read_csv(path("tables_dir") / "gold_overlap.csv")


if __name__ == "__main__":
    out = compute()
    out.to_csv(path("tables_dir") / "gold_overlap.csv", index=False, lineterminator="\n")
    print(f"{int(out['overlap'].sum())} of {len(out)} SmartBugs contracts overlap Dataset A "
          f"(exact {int(out['exact_match'].sum())}, near {int(out['near_duplicate'].sum())}, "
          f"address {int(out['same_address'].sum())}); by split of the match:")
    print(out[out["overlap"]]["matched_split"].value_counts().to_string())
