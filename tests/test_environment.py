"""
Environment checks (blueprint P1.1): configuration, folders and pinned dependencies.
"""

import re

from src.config import ROOT, load_config, path


def test_config_loads_with_required_sections():
    cfg = load_config()
    assert cfg["project"]["name"] == "muhafiz-sol"
    assert cfg["project"]["random_seed"] == 42
    for section in ("paths", "dataset", "labels", "preprocessing", "deduplication",
                    "features", "training", "explain"):
        assert section in cfg, section
    assert abs(sum(cfg["dataset"]["split_ratios"].values()) - 1.0) < 1e-9


def test_repository_structure_exists():
    for d in ("src", "tests", "app", "models", "notebooks", "docs", "reports"):
        assert (ROOT / d).is_dir(), d
    for key in ("raw_dir", "processed_dir", "gold_dir", "features_dir"):
        assert path(key).is_dir(), key


def test_requirements_are_pinned():
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    names = {re.split(r"==", ln)[0].lower() for ln in lines}
    assert {"pandas", "numpy", "scikit-learn", "datasketch", "streamlit", "pytest"} <= names
    assert all("==" in ln for ln in lines), "every requirement must be pinned"
