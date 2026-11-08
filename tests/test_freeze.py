"""
Phase 2 acceptance tests B2, B3 and B6 (blueprint 14.2).
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, path, read_json

MODELS_READY = (path("models_dir") / "thresholds.json").exists()
IS_GIT_CHECKOUT = (ROOT / ".git").exists()  # the submission zip has no .git folder and no tags
FEATURES_READY = (path("features_dir") / "X_train.npz").exists()


def test_b2_training_code_cannot_read_test_or_gold():
    from src.train import load_xy

    for split in ("test", "gold"):
        with pytest.raises(PermissionError):
            load_xy(split)
    with pytest.raises(PermissionError):
        load_xy("test", prefix="random_")


def test_b2_evaluation_refuses_without_tag(monkeypatch):
    from src import run_experiments

    class Fake:
        stdout, returncode = "", 1

    monkeypatch.setattr(run_experiments.subprocess, "run", lambda *a, **k: Fake())
    with pytest.raises(SystemExit):
        run_experiments.check_frozen()


@pytest.mark.skipif(not MODELS_READY or not IS_GIT_CHECKOUT,
                    reason="needs the git checkout with tag model-v1 (not part of the submission zip)")
def test_b2_tag_exists_and_thresholds_unchanged_since_freeze():
    from src.run_experiments import check_frozen

    assert check_frozen() != "untagged"


@pytest.mark.skipif(not (path("tables_dir") / "e1_test_summary.csv").exists(), reason="experiments not run")
def test_b6_baselines_in_every_table():
    for name, col in (("e1_test_summary.csv", "key"), ("e2_leakage.csv", "key"), ("e3_summary.csv", "key")):
        keys = set(pd.read_csv(path("tables_dir") / name)[col])
        assert {"prior", "logreg"} <= keys, name


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1" or not MODELS_READY,
                    reason="slow (retrains logistic regression); set RUN_SLOW=1")
def test_b3_logreg_retrain_reproduces_frozen_validation_score():
    from src.evaluate import pr_auc
    from src.models import LogRegModel
    from src.train import load_xy

    meta = read_json(path("models_dir") / "thresholds.json")
    sel = pd.read_csv(path("tables_dir") / "model_selection_val.csv").set_index("model")
    Xtr, Ytr = load_xy("train")
    Xva, Yva = load_xy("val")
    m = LogRegModel(**meta["all_models"]["logreg"]["params"]).fit(Xtr, Ytr)
    ap = float(np.nanmean(pr_auc(Yva, m.predict_proba(Xva))))
    assert abs(ap - sel.loc["logreg", "val_macro_pr_auc"]) < 1e-3


@pytest.mark.skipif(not FEATURES_READY, reason="feature matrices not built (python -m src.build_data)")
def test_b3_mlp_training_is_deterministic_for_a_fixed_seed():
    """B3 for the primary model family: two MLP runs with the same seed give identical predictions.
    (A short run on a subsample; the full 25-epoch retrain is too slow for the test suite.)"""
    import torch

    from src.models import MLPModel
    from src.train import load_xy

    torch.set_num_threads(1)
    Xtr, Ytr = load_xy("train")
    Xva, Yva = load_xy("val")
    Xs, Ys, Xv, Yv = Xtr[:3000], Ytr[:3000], Xva[:1000], Yva[:1000]
    runs = [MLPModel(max_epochs=2, seed=42, verbose=False, device="cpu").fit(Xs, Ys, Xv, Yv) for _ in range(2)]
    a, b = (r.predict_proba(Xv) for r in runs)
    assert np.array_equal(a, b)
    assert runs[0].history.val_pr_auc == runs[1].history.val_pr_auc
