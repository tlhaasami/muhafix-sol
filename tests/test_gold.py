"""
Tests for the SmartBugs gold set (blueprint P1.3, acceptance test A1).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.config import path
from src.preprocess import count_lines, split_lines

GOLD = path("gold_dir") / "smartbugs_gold.parquet"
MARKERS = ("@vulnerable_at_lines", "<yes>", "<report>", "@source", "@author")


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    if not GOLD.exists():
        # the gold parquet is committed; rebuilding it needs git and network access to GitHub
        try:
            from src.build_gold import build_smartbugs_gold

            build_smartbugs_gold()
        except Exception as exc:  # offline or no git
            pytest.skip(f"SmartBugs gold set missing and could not be rebuilt: {exc}")
    return pd.read_parquet(GOLD)


def test_gold_counts_match_vulnerabilities_json(gold):
    assert len(gold) == 143
    assert int(gold["n_vulnerabilities"].sum()) == 207
    assert gold["id"].is_unique


def test_gold_required_columns(gold):
    assert {"id", "category", "source_clean", "vulnerable_lines"} <= set(gold.columns)
    assert gold["vulnerable_lines"].map(len).min() >= 1


def test_a1_no_annotation_leakage(gold):
    """A1: no annotation marker survives cleaning (zero matches)."""
    for text in gold["source_clean"]:
        for marker in MARKERS:
            assert marker not in text


def test_annotated_lines_are_valid_and_line_count_preserved(gold):
    root = path("smartbugs_dir")
    for row in gold.itertuples():
        raw = (root / row.path).read_text(encoding="utf-8", errors="replace")
        assert count_lines(raw) == count_lines(row.source_clean) == row.n_lines
        assert all(1 <= ln <= row.n_lines for ln in row.vulnerable_lines)
        # the annotated code line itself is still present after cleaning
        clean_lines = split_lines(row.source_clean)
        assert any(clean_lines[ln - 1].strip() for ln in row.vulnerable_lines)
