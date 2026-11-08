"""
Muhafiz.sol - evaluation harness (blueprint P2.1, P2.4).

Every experiment stores results in ONE fixed long-format schema:

    model, split, class, metric, value

(``class`` is a class name, or ``macro`` / ``micro`` / ``all`` for aggregates;
bootstrap bounds are stored as separate metrics, e.g. ``recall_ci_low``.)

Functions take plain numpy arrays: ``y_true`` (n, C) of 0/1, ``y_prob`` (n, C)
of probabilities and ``thresholds`` (C,), so they can be unit-tested on tiny
hand-made arrays with known answers (acceptance test B1).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RESULT_COLUMNS = ["model", "split", "class", "metric", "value"]


# --------------------------------------------------------------------------- #
# basic counts and metrics
# --------------------------------------------------------------------------- #
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-class TP / FP / FN / TN (arrays of length C)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": (y_true & y_pred).sum(0),
        "fp": (~y_true & y_pred).sum(0),
        "fn": (y_true & ~y_pred).sum(0),
        "tn": (~y_true & ~y_pred).sum(0),
    }


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def fbeta(precision, recall, beta: float = 1.0):
    b2 = beta * beta
    return _safe_div((1 + b2) * np.asarray(precision) * np.asarray(recall),
                     b2 * np.asarray(precision) + np.asarray(recall))


def binarize(y_prob: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    return (np.asarray(y_prob) >= np.asarray(thresholds)[None, :]).astype(np.int8)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Average precision per class; NaN for classes without positives."""
    y_true = np.asarray(y_true)
    out = np.full(y_true.shape[1], np.nan)
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            out[c] = average_precision_score(y_true[:, c], y_prob[:, c])
    return out


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true).astype(bool) != np.asarray(y_pred).astype(bool)))


def per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float],
                      classes: Sequence[str]) -> pd.DataFrame:
    """One row per class with precision, recall, F1, F2, PR-AUC and counts."""
    y_pred = binarize(y_prob, thresholds)
    cc = confusion_counts(y_true, y_pred)
    precision = _safe_div(cc["tp"], cc["tp"] + cc["fp"])
    recall = _safe_div(cc["tp"], cc["tp"] + cc["fn"])
    df = pd.DataFrame({
        "class": list(classes),
        "precision": precision,
        "recall": recall,
        "f1": fbeta(precision, recall, 1.0),
        "f2": fbeta(precision, recall, 2.0),
        "pr_auc": pr_auc(y_true, y_prob),
        "support": np.asarray(y_true).sum(0),
        "prevalence": np.asarray(y_true).mean(0),
        "fpr": _safe_div(cc["fp"], cc["fp"] + cc["tn"]),
        "threshold": np.asarray(thresholds, dtype=float),
        **{k: v for k, v in cc.items()},
    })
    return df


def summary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    """Macro / micro averages, macro PR-AUC, Hamming loss, subset accuracy."""
    y_pred = binarize(y_prob, thresholds)
    cc = confusion_counts(y_true, y_pred)
    p = _safe_div(cc["tp"], cc["tp"] + cc["fp"])
    r = _safe_div(cc["tp"], cc["tp"] + cc["fn"])
    tp, fp, fn = cc["tp"].sum(), cc["fp"].sum(), cc["fn"].sum()
    mp = float(_safe_div(tp, tp + fp))
    mr = float(_safe_div(tp, tp + fn))
    aps = pr_auc(y_true, y_prob)
    return {
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(fbeta(p, r, 1).mean()),
        "macro_f2": float(fbeta(p, r, 2).mean()),
        "micro_precision": mp,
        "micro_recall": mr,
        "micro_f1": float(fbeta(mp, mr, 1)),
        "macro_pr_auc": float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan"),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": float(np.mean((np.asarray(y_true) == y_pred).all(1))),
        "label_accuracy": 1.0 - hamming_loss(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, iterations: int = 1000,
                 alpha: float = 0.05, seed: int = 42, groups: Optional[Sequence] = None) -> Tuple[float, float]:
    """Percentile bootstrap interval of ``stat(idx)`` where idx resamples 0..n-1.

    With ``groups`` (one id per item) whole groups are resampled (cluster bootstrap), so near-duplicate
    contracts that share a clone group are not treated as independent evidence."""
    rng = np.random.default_rng(seed)
    members = None
    if groups is not None:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        order = np.argsort(inv, kind="mergesort")
        members = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)
    vals = []
    for _ in range(iterations):
        if members is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def recall_with_ci(y_true_col: np.ndarray, y_pred_col: np.ndarray, iterations: int = 1000,
                   seed: int = 42) -> Tuple[float, float, float, int]:
    """Recall of one class with a bootstrap interval over the positive contracts."""
    pos = np.flatnonzero(np.asarray(y_true_col) == 1)
    if len(pos) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    hits = np.asarray(y_pred_col)[pos].astype(float)
    lo, hi = bootstrap_ci(lambda idx: hits[idx].mean(), len(hits), iterations, seed=seed)
    return float(hits.mean()), lo, hi, int(len(pos))


def metric_ci(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float],
              key: str, iterations: int = 1000, seed: int = 42,
              groups: Optional[Sequence] = None) -> Tuple[float, float]:
    """Bootstrap interval for a summary metric (e.g. 'macro_f1') over contracts (or clone groups)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)

    def stat(idx):
        return summary_metrics(y_true[idx], y_prob[idx], thresholds)[key]

    return bootstrap_ci(stat, len(y_true), iterations, seed=seed, groups=groups)


# --------------------------------------------------------------------------- #
# thresholds, risk, calibration
# --------------------------------------------------------------------------- #
def choose_threshold(y_true_col: np.ndarray, prob_col: np.ndarray, beta: float = 2.0,
                     min_precision: float = 0.2) -> Tuple[float, Dict[str, float]]:
    """Threshold maximising F-beta on validation, subject to precision >= guard.

    Candidate thresholds are the distinct predicted scores. If no threshold
    reaches the precision guard, the F-beta maximiser without the guard is used
    (and reported as such).
    """
    y = np.asarray(y_true_col).astype(int)
    p = np.asarray(prob_col, dtype=float)
    if y.sum() == 0:
        return 0.5, {"fbeta": 0.0, "precision": 0.0, "recall": 0.0, "guard_met": False}
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # only evaluate at the last index of each run of equal scores
    last = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    tp, fp, thr = tp[last], fp[last], p_sorted[last]
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    f = fbeta(precision, recall, beta)
    ok = precision >= min_precision
    guard = bool(ok.any())
    i = int(np.argmax(np.where(ok, f, -1))) if guard else int(np.argmax(f))
    return float(thr[i]), {"fbeta": float(f[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "guard_met": guard}


def choose_thresholds(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str],
                      beta: float = 2.0, min_precision: float = 0.2) -> Tuple[np.ndarray, pd.DataFrame]:
    rows, thr = [], []
    for c, name in enumerate(classes):
        t, info = choose_threshold(y_true[:, c], y_prob[:, c], beta, min_precision)
        thr.append(t)
        rows.append({"class": name, "threshold": t, **info})
    return np.array(thr), pd.DataFrame(rows)


def high_cutoffs(thresholds: Sequence[float], margin: float) -> np.ndarray:
    """Stricter confidence level for 'high': t + margin * (1 - t)."""
    t = np.asarray(thresholds, dtype=float)
    return t + margin * (1.0 - t)


def risk_level(probs: Sequence[float], thresholds: Sequence[float], margin: float) -> str:
    """Documented rule (blueprint 8.4): low if no class reaches its threshold,
    high if any class reaches the stricter level, medium otherwise."""
    p = np.asarray(probs, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    if not (p >= t).any():
        return "low"
    return "high" if (p >= high_cutoffs(t, margin)).any() else "medium"


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str],
                      bins: int = 10) -> pd.DataFrame:
    """Reliability-curve data: mean predicted vs observed frequency per bin."""
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for c, name in enumerate(classes):
        idx = np.clip(np.digitize(y_prob[:, c], edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if m.any():
                rows.append({"class": name, "bin": b, "mean_predicted": float(y_prob[m, c].mean()),
                             "observed": float(y_true[m, c].mean()), "count": int(m.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true_col: np.ndarray, prob_col: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob_col, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(prob_col[m].mean() - y_true_col[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# long-format results
# --------------------------------------------------------------------------- #
def to_long(model: str, split: str, per_class: Optional[pd.DataFrame] = None,
            summary: Optional[Dict[str, float]] = None,
            metrics: Iterable[str] = ("precision", "recall", "f1", "f2", "pr_auc", "support", "fpr",
                                      "threshold")) -> pd.DataFrame:
    rows: List[dict] = []
    if per_class is not None:
        for _, r in per_class.iterrows():
            for m in metrics:
                if m in r:
                    rows.append({"model": model, "split": split, "class": r["class"], "metric": m,
                                 "value": float(r[m])})
    if summary:
        for k, v in summary.items():
            agg, _, name = k.partition("_")
            cls = agg if agg in ("macro", "micro") else "all"
            rows.append({"model": model, "split": split, "class": cls,
                         "metric": name if cls != "all" else k, "value": float(v)})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def evaluate(model: str, split: str, y_true: np.ndarray, y_prob: np.ndarray,
             thresholds: Sequence[float], classes: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    pc = per_class_metrics(y_true, y_prob, thresholds, classes)
    sm = summary_metrics(y_true, y_prob, thresholds)
    return to_long(model, split, pc, sm), pc, sm
