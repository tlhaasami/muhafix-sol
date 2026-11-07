"""
Acceptance Test Suite for SmartBugs Gold Set (Work Package P1.3 and Test A1)
Owner: Talha Sami (24I-3118)
"""

import os
import json
import pytest

GOLD_JSON = "data/gold/smartbugs_gold.json"


@pytest.fixture(scope="module")
def gold_data():
    if not os.path.exists(GOLD_JSON):
        from src.build_gold import build_smartbugs_gold
        build_smartbugs_gold()
    with open(GOLD_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_gold_dataset_non_empty(gold_data):
    assert len(gold_data) > 0, "SmartBugs gold set is empty"
    assert len(gold_data) == 143, f"Expected 143 contracts, found {len(gold_data)}"


def test_gold_required_fields(gold_data):
    required_keys = {"id", "category", "source_clean", "vulnerable_lines"}
    for item in gold_data:
        assert required_keys.issubset(item.keys()), f"Item {item.get('id')} is missing required keys"


def test_a1_no_annotation_leakage(gold_data):
    """
    Acceptance Test A1: Zero matches for annotation markers in cleaned gold contracts.
    """
    forbidden_markers = ["@vulnerable_at_lines", "<yes>", "<report>"]
    for item in gold_data:
        clean_text = item["source_clean"]
        for marker in forbidden_markers:
            assert marker not in clean_text, (
                f"Leakage detected in contract {item['id']}: found marker '{marker}'"
            )


def test_gold_line_number_preservation(gold_data):
    """
    Ensure that line numbers were preserved so line-level annotations match correctly.
    """
    for item in gold_data:
        assert item["source_raw_lines"] == item["source_clean_lines"], (
            f"Line count mismatch in {item['id']}: "
            f"raw={item['source_raw_lines']} vs clean={item['source_clean_lines']}"
        )
