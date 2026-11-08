"""
Helpers shared by the report, slide and model-card generators: a ``Facts``
object that reads every number from the result tables (so the report, slides
and app always agree) and small python-docx convenience functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import path, read_json


def _read(name: str) -> Optional[pd.DataFrame]:
    p = path("tables_dir") / name
    return pd.read_csv(p) if p.exists() else None


class Facts:
    """Every number used in documents, loaded from reports/tables and build artefacts."""

    def __init__(self):
        def opt(p):
            return read_json(p) if p.exists() else {}

        self.summary = read_json(path("processed_dir") / "build_summary.json")
        self.label_map = read_json(path("models_dir") / "label_map.json")
        self.thresholds = opt(path("models_dir") / "thresholds.json")       # after Phase 2
        self.feature_stats = opt(path("features_dir") / "feature_stats.json")
        self.classes = self.label_map["target_classes"]
        self.primary = self.thresholds.get("primary", "logreg")
        t = {}
        for name in ["dataset_funnel", "label4_audit", "detector_frequency", "family_distribution",
                     "label_distribution", "length_stats", "solidity_versions", "smartbugs_counts",
                     "smartbugs_pragmas", "validation_grid", "model_selection_val", "thresholds_val", "mlp_seeds",
                     "e1_test_summary", "e1_mlp_seeds_test", "e2_leakage", "e3_smartbugs_recall",
                     "e3_smartbugs_all_models", "e3_summary", "e3b_hit_rate", "e4_ablations", "e5_latency",
                     "e5_error_slices"]:
            t[name] = _read(f"{name}.csv")
        self.t = t
        pc = path("tables_dir") / f"e1_per_class_{self.primary}.csv"
        self.per_class_primary = pd.read_csv(pc) if pc.exists() else None
        cs = path("tables_dir") / "e5_case_studies.json"
        self.cases = json.loads(cs.read_text(encoding="utf-8")) if cs.exists() else []
        rates = path("tables_dir") / "e2_leakage_rates.json"
        self.leak_rates = json.loads(rates.read_text()) if rates.exists() else {}

    # convenience accessors
    @property
    def funnel(self) -> Dict[str, int]:
        return self.summary["funnel"]

    @property
    def dedup(self) -> Dict[str, Any]:
        return self.summary["dedup"]

    @property
    def sizes(self) -> Dict[str, int]:
        return self.summary["split_sizes"]

    def e1(self, key: str) -> pd.Series:
        df = self.t["e1_test_summary"]
        return df[df["key"] == key].iloc[0]


MODEL_NAMES = {"prior": "Majority / prior baseline", "logreg": "Logistic regression", "tree": "Decision tree",
               "forest": "Random forest", "knn": "K-nearest neighbours", "mlp": "MLP neural network"}


# --------------------------------------------------------------------------- #
# python-docx helpers
# --------------------------------------------------------------------------- #
def fmt(v, nd: int = 3) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (bool,)) or type(v).__name__ == "bool_":
        return "yes" if v else "no"
    try:
        if pd.isna(v):
            return "-"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and abs(v) >= 10):
        return f"{int(v):,}"
    return f"{v:.{nd}f}"


def pct(v, nd: int = 1) -> str:
    return "-" if v is None or pd.isna(v) else f"{100 * v:.{nd}f}%"


def model_lc(name: str) -> str:
    """Model name for running text: 'Logistic regression' -> 'logistic regression', but 'MLP neural network' stays."""
    first = name.split()[0] if name else ""
    return name if first.isupper() else name[:1].lower() + name[1:]
