"""
Acceptance test B1: the evaluation harness on hand-made arrays with known answers.
"""

import numpy as np
import pytest

from src.evaluate import (
    RESULT_COLUMNS,
    binarize,
    bootstrap_ci,
    calibration_table,
    choose_threshold,
    choose_thresholds,
    confusion_counts,
    evaluate,
    expected_calibration_error,
    fbeta,
    hamming_loss,
    high_cutoffs,
    per_class_metrics,
    pr_auc,
    recall_with_ci,
    risk_level,
    summary_metrics,
)

Y = np.array([[1, 0], [1, 1], [0, 0], [0, 1], [1, 0]])
P = np.array([[0.9, 0.2], [0.4, 0.8], [0.6, 0.1], [0.2, 0.3], [0.7, 0.6]])
T = np.array([0.5, 0.5])


def test_confusion_counts_known_values():
    cc = confusion_counts(Y, binarize(P, T))
    # class 0: preds [1,0,1,0,1] vs [1,1,0,0,1] -> tp2 fp1 fn1 tn1
    assert cc["tp"].tolist() == [2, 1] and cc["fp"].tolist() == [1, 1]
    assert cc["fn"].tolist() == [1, 1] and cc["tn"].tolist() == [1, 2]


def test_per_class_precision_recall_f1_f2():
    pc = per_class_metrics(Y, P, T, ["a", "b"])
    a = pc.iloc[0]
    assert a.precision == pytest.approx(2 / 3) and a.recall == pytest.approx(2 / 3)
    assert a.f1 == pytest.approx(2 / 3) and a.f2 == pytest.approx(2 / 3)
    b = pc.iloc[1]
    assert b.precision == pytest.approx(0.5) and b.recall == pytest.approx(0.5)
    assert fbeta(0.5, 1.0, 2.0) == pytest.approx(5 * 0.5 / (4 * 0.5 + 1.0))


def test_pr_auc_matches_hand_computation():
    # class 0 ranking by score: .9(1) .7(1) .6(0) .4(1) .2(0) -> AP = (1 + 1 + 3/4)/3
    assert pr_auc(Y, P)[0] == pytest.approx((1 + 1 + 0.75) / 3)
    assert np.isnan(pr_auc(np.zeros((3, 1)), np.zeros((3, 1)))[0])


def test_summary_macro_micro_hamming():
    s = summary_metrics(Y, P, T)
    assert s["macro_precision"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert s["micro_precision"] == pytest.approx(3 / 5)  # tp=3, fp=2
    assert s["micro_recall"] == pytest.approx(3 / 5)     # tp=3, fn=2
    assert s["hamming_loss"] == pytest.approx(4 / 10)
    assert hamming_loss(Y, Y) == 0.0
    assert s["subset_accuracy"] == pytest.approx(1 / 5)  # only the last row is exact


def test_threshold_maximises_f2_with_guard():
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    p = np.array([.9, .8, .7, .6, .5, .4, .3, .1])
    t, info = choose_threshold(y, p, beta=2.0, min_precision=0.0)
    assert t == pytest.approx(0.3) and info["recall"] == 1.0   # recall-favouring choice
    t2, info2 = choose_threshold(y, p, beta=2.0, min_precision=0.9)
    assert t2 == pytest.approx(0.8) and info2["precision"] == 1.0 and info2["guard_met"]
    thr, table = choose_thresholds(Y, P, ["a", "b"])
    assert len(thr) == 2 and list(table["class"]) == ["a", "b"]


def test_risk_rule():
    t = [0.5, 0.4]
    assert risk_level([0.1, 0.2], t, 0.5) == "low"
    assert risk_level([0.6, 0.1], t, 0.5) == "medium"
    assert risk_level([0.1, 0.71], t, 0.5) == "high"   # 0.4 + 0.5 * 0.6 = 0.70
    assert high_cutoffs(t, 0.5).tolist() == pytest.approx([0.75, 0.70])


def test_bootstrap_interval_contains_point_and_is_reproducible():
    hits = np.array([1, 1, 1, 0] * 25, dtype=float)
    lo, hi = bootstrap_ci(lambda i: hits[i].mean(), len(hits), 500, seed=1)
    assert lo < 0.75 < hi and (lo, hi) == bootstrap_ci(lambda i: hits[i].mean(), len(hits), 500, seed=1)
    r, lo, hi, n = recall_with_ci(np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), 200)
    assert r == pytest.approx(2 / 3) and n == 3 and lo <= r <= hi


def test_calibration_and_ece():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0
    tab = calibration_table(y[:, None], p[:, None], ["a"], bins=2)
    assert tab["count"].sum() == 4


def test_long_format_schema():
    long, pc, sm = evaluate("lr", "test", Y, P, T, ["a", "b"])
    assert list(long.columns) == RESULT_COLUMNS
    assert {"precision", "recall", "f1", "pr_auc"} <= set(long["metric"])
    assert ((long["class"] == "macro") & (long["metric"] == "f1")).any()
