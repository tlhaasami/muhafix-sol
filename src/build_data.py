"""
Muhafiz.sol - one-command Phase 1 data build (blueprint P1.8, acceptance test A6).

    python -m src.build_data            # run every stage (cached stages are skipped)
    python -m src.build_data --force    # rebuild everything
    python -m src.build_data --stage features

Stages
  1 ingest    download Dataset A shards (src.ingest_slither)
  2 gold      SmartBugs gold set (src.build_gold)
  3 tokens    clean / normalise / tokenise every contract, long-file policy,
              exact hash and MinHash signature (parallel)
  4 splits    filter, label families, rare-class rule, exact + near-duplicate
              groups, group-aware split, random split (E2), label_map.json
  5 features  TF-IDF fitted on TRAIN only; matrices for every split + gold
  6 reports   tables and figures for the data card (src.eda)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.config import ROOT, load_config, path, read_json, seed, write_json
from src.preprocess import clean_source, count_lines, normalise, tokenize, truncate_tokens

# [ \t]* (not \s*): with re.M, \s* would re-scan long blank runs (blanked comments) from
# every line start inside them - quadratic on million-character flattened files.
_CONTRACT_DECL = re.compile(r"^[ \t]*(?:abstract[ \t]+)?contract[ \t]+([A-Za-z_$][\w$]*)", re.M)
_PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);")
_VERSION = re.compile(r"0\.(\d+)\.(\d+)")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def solidity_minor(pragma: str) -> str:
    """'^0.4.24' -> '0.4'; used to compare compiler generations."""
    m = _VERSION.search(pragma or "")
    return f"0.{m.group(1)}" if m else "unknown"


def main_contract_snippet(clean: str, max_lines: int = 14, max_chars: int = 700) -> tuple[str, str]:
    """Name of the last declared contract (the main one in a flattened file) and
    a short preview starting at its declaration."""
    matches = list(_CONTRACT_DECL.finditer(clean))
    if not matches:
        return "", ""
    m = matches[-1]
    lines = [ln.rstrip() for ln in clean[m.start():].splitlines() if ln.strip()]
    return m.group(1), "\n".join(lines[:max_lines])[:max_chars]


def _token_job(addresses: List[str], sources: List[str], max_tokens: int, head_fraction: float,
               k: int, num_perm: int) -> List[dict]:
    from src.splits import dedup_tokens, exact_hash, minhash_signature

    out = []
    for addr, src in zip(addresses, sources):
        src = src or ""
        clean = clean_source(src)
        toks = tokenize(normalise(clean))
        kept, truncated = truncate_tokens(toks, max_tokens, head_fraction)
        dtoks = dedup_tokens(kept)
        name, snippet = main_contract_snippet(clean)
        pm = _PRAGMA.search(clean)
        out.append({
            "address": addr,
            "tokens": " ".join(kept),
            "n_tokens": len(toks),
            "truncated": truncated,
            "n_lines": count_lines(src),
            "n_chars": len(src),
            "pragma": pm.group(1).strip()[:40] if pm else "none",
            "contract_name": name,
            "snippet": snippet,
            "exact_hash": exact_hash(dtoks),
            "signature": minhash_signature(dtoks, k, num_perm).tobytes(),
        })
    return out


def split_hash(addresses: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(addresses)).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# stage 3: tokens
# --------------------------------------------------------------------------- #
def stage_tokens(force: bool = False) -> Path:
    from joblib import Parallel, delayed

    cfg = load_config()
    out = path("interim_dir") / "tokens.parquet"
    if out.exists() and not force:
        _log(f"tokens: {out.name} exists - skipped")
        return out
    src = path("interim_dir") / "slither_contracts.parquet"
    pre, ded = cfg["preprocessing"], cfg["deduplication"]
    pf = pq.ParquetFile(src)
    writer = None
    tmp = out.with_suffix(".parquet.part")
    done, t0 = 0, time.time()
    with Parallel(n_jobs=-1) as par:
        for batch in pf.iter_batches(batch_size=3000, columns=["address", "source_code"]):
            addrs = batch.column("address").to_pylist()
            srcs = batch.column("source_code").to_pylist()
            step = 50
            results = par(delayed(_token_job)(addrs[i:i + step], srcs[i:i + step], pre["max_tokens"],
                                              pre["head_fraction"], ded["shingle_size"],
                                              ded["minhash_num_perm"])
                          for i in range(0, len(addrs), step))
            table = pa.Table.from_pylist([r for part in results for r in part])
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            done += len(addrs)
            _log(f"tokens: {done:,}/{pf.metadata.num_rows:,} ({time.time() - t0:.0f}s)")
    writer.close()
    tmp.replace(out)
    return out


# --------------------------------------------------------------------------- #
# stage 4: labels, de-duplication, splits
# --------------------------------------------------------------------------- #
def stage_splits(force: bool = False) -> Path:
    from src.labels import SMARTBUGS_MAPPING, encode, families_for, family_map, select_classes
    from src.splits import cluster, group_stratified_split, leakage_rate, random_split

    cfg = load_config()
    proc = path("processed_dir")
    out = proc / "dataset.parquet"
    if out.exists() and not force:
        _log(f"splits: {out.name} exists - skipped")
        return out

    meta = pd.read_parquet(path("interim_dir") / "slither_contracts.parquet",
                           columns=["address", "success", "error", "detectors", "hf_config", "hf_split"])
    tok = pd.read_parquet(path("interim_dir") / "tokens.parquet",
                          columns=["address", "n_tokens", "truncated", "n_lines", "n_chars", "pragma",
                                   "contract_name", "snippet", "exact_hash", "signature"])
    df = pd.concat([meta.reset_index(drop=True), tok.drop(columns="address").reset_index(drop=True)], axis=1)
    df["row"] = np.arange(len(df))
    funnel = {"rows_in_raw_shards": len(df)}

    df = df[~df["address"].duplicated(keep="first")]
    funnel["after_dropping_duplicate_addresses"] = len(df)
    df = df[df["success"]]
    funnel["after_dropping_failed_slither_runs"] = len(df)
    df = df[df["n_tokens"] >= 20]
    funnel["after_dropping_empty_or_tiny_sources"] = len(df)
    df = df.reset_index(drop=True)

    upstream = read_json(path("slither_raw_dir") / "label_mappings.json")
    fmap = family_map(upstream)
    df["families"] = [families_for(d, fmap) for d in df["detectors"]]

    # exact + near duplicate groups over the filtered pool
    sigs = np.frombuffer(b"".join(df["signature"].tolist()), dtype=np.uint64).reshape(len(df), -1)
    _log(f"splits: clustering {len(df):,} contracts")
    groups, dstats = cluster(df["exact_hash"].tolist(), sigs, cfg["deduplication"]["jaccard_threshold"])
    df["group"] = groups
    is_rep = ~df["exact_hash"].duplicated(keep="first")
    fam_key = df["families"].map(lambda f: "|".join(f))
    conflicts = df.assign(k=fam_key).groupby("exact_hash")["k"].nunique()
    dstats["exact_groups_with_label_conflicts"] = int((conflicts > 1).sum())
    dstats["exact_groups_with_duplicates"] = int((df.groupby("exact_hash").size() > 1).sum())
    df["exact_group_size"] = df.groupby("exact_hash")["address"].transform("size")
    df = df[is_rep].reset_index(drop=True)
    funnel["after_exact_deduplication"] = len(df)

    # rare-class rule on the de-duplicated pool
    lab_cfg = cfg["labels"]
    ratios = cfg["dataset"]["split_ratios"]
    sel = select_classes(df["families"].tolist(), lab_cfg["min_prevalence"],
                         lab_cfg["min_train_positives"], ratios["train"])
    classes = sel["classes"]
    Y = encode(df["families"].tolist(), classes, sel["merged_into_other"])
    for i, c in enumerate(classes):
        df[f"y_{c}"] = Y[:, i]
    df["labels"] = [[c for c, v in zip(classes, row) if v] for row in Y]

    _, dense_group = np.unique(df["group"].to_numpy(), return_inverse=True)
    df["group"] = dense_group
    split = group_stratified_split(dense_group, Y, (ratios["train"], ratios["val"], ratios["test"]), seed(),
                                   cfg["deduplication"]["max_eval_group_fraction"])
    sizes = tuple(int((split == k).sum()) for k in range(3))
    rsplit = random_split(len(df), sizes, seed())
    names = np.array(["train", "val", "test"])
    df["split"] = names[split]
    df["split_random"] = names[rsplit]
    dstats["random_split_leakage_rate"] = leakage_rate(dense_group, rsplit)
    dstats["group_split_leakage_rate"] = leakage_rate(dense_group, split)
    gsize = np.bincount(dense_group)
    dstats["groups_after_exact_dedup"] = int(len(gsize))
    dstats["largest_group_after_exact_dedup"] = int(gsize.max())
    giant = gsize > cfg["deduplication"]["max_eval_group_fraction"] * len(df)
    dstats["giant_groups_forced_to_train"] = int(giant.sum())
    dstats["contracts_in_giant_groups"] = int(gsize[giant].sum())
    dstats["largest_group_in_val_or_test"] = int(max(np.bincount(dense_group[split != 0]).max(), 0))

    # attach token strings (row-aligned with the interim file)
    tokens = pd.read_parquet(path("interim_dir") / "tokens.parquet", columns=["tokens"])["tokens"]
    df["tokens"] = tokens.iloc[df["row"].to_numpy()].to_numpy()
    keep_cols = (["address", "group", "split", "split_random", "labels", "families", "detectors",
                  "hf_config", "hf_split", "n_tokens", "truncated", "n_lines", "n_chars", "pragma",
                  "contract_name", "snippet", "exact_group_size"] + [f"y_{c}" for c in classes] + ["tokens"])
    df = df[keep_cols]
    proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    df[["address", "group", "split", "split_random"]].to_csv(
        proc / "split_manifest.csv.gz", index=False, compression="gzip", lineterminator="\n")

    hashes = {s: split_hash(df.loc[df["split"] == s, "address"]) for s in ["train", "val", "test"]}
    hashes.update({f"random_{s}": split_hash(df.loc[df["split_random"] == s, "address"])
                   for s in ["train", "val", "test"]})

    label_map = {
        "version": cfg["project"]["version"],
        "target_classes": classes,
        "class_to_id": {c: i for i, c in enumerate(classes)},
        "merged_into_other": sel["merged_into_other"],
        "excluded_families": sel["excluded"],
        "family_stats_dedup_pool": sel["family_stats"],
        "detector_to_family": fmap,
        "overrides_vs_upstream": {"unchecked-send": {"upstream": upstream.get("unchecked-send"),
                                                     "ours": "unchecked-calls"}},
        "smartbugs_category_mapping": {
            k: {"class": v[0] if v[0] in classes else None,
                "status": v[1] if (v[0] is None or v[0] in classes) else "not evaluated",
                "reason": v[2] if (v[0] is None or v[0] in classes) else f"class {v[0]} not retained"}
            for k, v in SMARTBUGS_MAPPING.items()},
        "rule": {"min_prevalence": lab_cfg["min_prevalence"],
                 "min_train_positives": lab_cfg["min_train_positives"]},
    }
    write_json(label_map, path("models_dir") / "label_map.json")
    write_json(label_map, ROOT / "data" / "label_map.json")  # copy next to the data (P1.6)

    summary = {"funnel": funnel, "dedup": dstats, "split_sizes": dict(zip(["train", "val", "test"], sizes)),
               "split_hashes": hashes, "classes": classes}
    write_json(summary, proc / "build_summary.json")
    _log(f"splits: {sizes} classes={classes}")
    return out


# --------------------------------------------------------------------------- #
# stage 5: features
# --------------------------------------------------------------------------- #
def load_split(name: str, column: str = "split") -> pd.DataFrame:
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet")
    return df[df[column] == name].reset_index(drop=True)


def gold_token_docs(gold: pd.DataFrame, max_tokens: int, head_fraction: float) -> List[str]:
    docs = []
    for text in gold["source_clean"]:
        toks, _ = truncate_tokens(tokenize(normalise(text)), max_tokens, head_fraction)
        docs.append(" ".join(toks))
    return docs


def stage_features(force: bool = False, ids: bool | None = None) -> Path:
    from src.features import FeaturePipeline

    cfg = load_config()
    fcfg, pre = cfg["features"], cfg["preprocessing"]
    vec_path = path("models_dir") / "vectorizer.joblib"
    feat = path("features_dir")
    if vec_path.exists() and (feat / "X_train.npz").exists() and not force:
        _log("features: exist - skipped")
        return vec_path
    ids = bool(fcfg.get("normalise_identifiers", False)) if ids is None else ids
    df = pd.read_parquet(path("processed_dir") / "dataset.parquet")
    classes = read_json(path("models_dir") / "label_map.json")["target_classes"]
    train = df[df["split"] == "train"]
    rng = np.random.default_rng(seed())
    n_vocab = min(int(fcfg["vocab_fit_docs"]), len(train))
    vocab_docs = train["tokens"].to_numpy()[rng.choice(len(train), n_vocab, replace=False)]

    t0 = time.time()
    pipe = FeaturePipeline(ngram_range=tuple(fcfg["ngram_range"]), max_features=fcfg["max_features"],
                           min_df=fcfg["min_df"], max_df=fcfg["max_df"], sublinear_tf=fcfg["sublinear_tf"],
                           ids=ids, svd_components=fcfg["svd_components"], random_state=seed())
    pipe.fit(train["tokens"].tolist(), vocab_docs=list(vocab_docs))
    _log(f"features: fitted on {len(train):,} train docs in {time.time() - t0:.0f}s {pipe.vocab_stats}")
    pipe.save(vec_path)

    feat.mkdir(parents=True, exist_ok=True)
    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet")
    blocks = {f"{s}": df[df["split"] == s] for s in ["train", "val", "test"]}
    blocks.update({f"random_{s}": df[df["split_random"] == s] for s in ["train", "val", "test"]})
    for name, part in blocks.items():
        X = pipe.transform(part["tokens"].tolist())
        sp.save_npz(feat / f"X_{name}.npz", X)
        np.save(feat / f"Y_{name}.npy", part[[f"y_{c}" for c in classes]].to_numpy(np.int8))
        part[["address"]].to_csv(feat / f"rows_{name}.csv", index=False, lineterminator="\n")
        if name in ("train", "val", "test"):
            np.save(feat / f"Xsvd_{name}.npy", pipe.transform_svd(X))
    Xg = pipe.transform(gold_token_docs(gold, pre["max_tokens"], pre["head_fraction"]))
    sp.save_npz(feat / "X_gold.npz", Xg)
    np.save(feat / "Xsvd_gold.npy", pipe.transform_svd(Xg))

    sample = df[df["split"] == "test"]["tokens"].iloc[0]
    stats = {**pipe.vocab_stats, "normalise_identifiers": ids,
             "single_contract_transform_ms": pipe.benchmark_single(sample),
             "fit_seconds": round(time.time() - t0, 1),
             "svd_explained_variance": float(pipe.svd.explained_variance_ratio_.sum())}
    write_json(stats, feat / "feature_stats.json")
    _log(f"features: done {stats}")
    return vec_path


# --------------------------------------------------------------------------- #
def run(stages: List[str], force: bool = False) -> None:
    from src.build_gold import build_smartbugs_gold
    from src.ingest_slither import ingest

    t0 = time.time()
    if "ingest" in stages:
        ingest(force=False)
    if "gold" in stages:
        _log(f"gold: {build_smartbugs_gold()}")
    if "tokens" in stages:
        stage_tokens(force)
    if "splits" in stages:
        stage_splits(force)
    if "features" in stages:
        stage_features(force)
    if "reports" in stages:
        from src.eda import build_reports

        build_reports()
    _log(f"build_data finished in {time.time() - t0:.0f}s")


STAGES = ["ingest", "gold", "tokens", "splits", "features", "reports"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, action="append", help="run only these stages")
    ap.add_argument("--force", action="store_true", help="rebuild cached stages")
    a = ap.parse_args()
    run(a.stage or STAGES, a.force)
