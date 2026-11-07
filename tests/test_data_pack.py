"""
Phase 1 acceptance tests A1-A6 on the REAL data pack (blueprint 13.2).
Skipped when the data pack has not been built (python -m src.build_data).
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_config, path, read_json, seed

DATASET = path("processed_dir") / "dataset.parquet"
pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="data pack not built")


@pytest.fixture(scope="module")
def ds():
    return pd.read_parquet(DATASET, columns=["address", "group", "split", "split_random", "tokens"]
                           + [f"y_{c}" for c in read_json(path("models_dir") / "label_map.json")["target_classes"]])


@pytest.fixture(scope="module")
def summary():
    return read_json(path("processed_dir") / "build_summary.json")


def test_a1_no_annotation_text_reaches_any_model_input(ds):
    """A1: SmartBugs markers never appear in gold text or in any training token stream."""
    gold = pd.read_parquet(path("gold_dir") / "smartbugs_gold.parquet")
    for text in gold["source_clean"]:
        assert "@vulnerable_at_lines" not in text and "<yes>" not in text and "<report>" not in text
    from src.build_data import gold_token_docs

    docs = gold_token_docs(gold, 20000, 0.25)
    assert not any("yes > < report" in d or "vulnerable_at_lines" in d for d in docs)
    sample = ds["tokens"].sample(2000, random_state=seed())
    assert not sample.str.contains("vulnerable_at_lines", regex=False).any()


def test_a2_no_group_in_two_splits(ds, summary):
    groups = {s: set(ds.loc[ds.split == s, "group"]) for s in ("train", "val", "test")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert summary["dedup"]["group_split_leakage_rate"] == 0.0
    assert ds["address"].is_unique


def test_a3_line_count_preserved_on_1000_dataset_a_contracts():
    from src.preprocess import clean_source, count_lines

    src = pd.read_parquet(path("interim_dir") / "slither_contracts.parquet", columns=["source_code"])
    sample = src["source_code"].sample(1000, random_state=seed())
    for text in sample:
        assert count_lines(clean_source(text)) == count_lines(text)


def test_a4_split_is_reproducible(ds, summary):
    """Re-running the split algorithm with seed 42 reproduces the stored split hashes."""
    from src.splits import group_stratified_split

    classes = read_json(path("models_dir") / "label_map.json")["target_classes"]
    r = load_config()["dataset"]["split_ratios"]
    split = group_stratified_split(ds["group"].to_numpy(), ds[[f"y_{c}" for c in classes]].to_numpy(),
                                   (r["train"], r["val"], r["test"]), seed(),
                                   load_config()["deduplication"]["max_eval_group_fraction"])
    names = np.array(["train", "val", "test"])[split]
    assert (names == ds["split"].to_numpy()).all()
    for s in ("train", "val", "test"):
        h = hashlib.sha256("\n".join(sorted(ds.loc[ds.split == s, "address"])).encode()).hexdigest()
        assert h == summary["split_hashes"][s]


def test_a5_every_class_has_positives_in_every_split(ds):
    ycols = [c for c in ds.columns if c.startswith("y_")]
    for col in ("split", "split_random"):
        for s in ("train", "val", "test"):
            assert (ds.loc[ds[col] == s, ycols].sum() > 0).all(), (col, s)


def test_split_sizes_within_tolerance(ds):
    frac = ds["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["val"] - 0.15) < 0.03 and abs(frac["test"] - 0.15) < 0.03
    assert (ds["split_random"].value_counts().sort_index() == ds["split"].value_counts().sort_index()).all()


def test_a6_one_command_build_entry_point_exists():
    from src import build_data

    assert callable(build_data.run) and set(build_data.STAGES) >= {"ingest", "gold", "tokens", "splits", "features"}
    assert (ROOT / "data" / "processed" / "split_manifest.csv.gz").exists()


def test_vectorizer_reproduces_stored_vector():
    """P1.7 done-when: transforming a contract reproduces its stored vector."""
    import scipy.sparse as sp

    from src.features import FeaturePipeline

    vec = FeaturePipeline.load(path("models_dir") / "vectorizer.joblib")
    X = sp.load_npz(path("features_dir") / "X_test.npz")
    rows = pd.read_csv(path("features_dir") / "rows_test.csv")
    ds = pd.read_parquet(DATASET, columns=["address", "tokens"]).set_index("address")
    for i in (0, 7, 123):
        again = vec.transform([ds.loc[rows.address[i], "tokens"]])
        assert np.allclose(again.toarray(), X[i].toarray(), atol=1e-6)
