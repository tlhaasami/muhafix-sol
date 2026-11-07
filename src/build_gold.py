"""
Muhafiz.sol - SmartBugs Gold Set Builder
Phase 1: Work Package P1.3
Owner: Talha Sami (24I-3118)

Loads curated contracts from smartbugs-curated, extracts line annotations,
cleans the contracts using line-preserving comment stripping,
and generates the gold evaluation dataset.
"""

import os
import json
import re
from typing import Dict, List, Any
from src.preprocess import clean_source


def build_smartbugs_gold(
    smartbugs_dir: str = "data/raw/smartbugs-curated",
    output_parquet: str = "data/gold/smartbugs_gold.parquet",
    output_json: str = "data/gold/smartbugs_gold.json",
    counts_csv: str = "reports/tables/smartbugs_counts.csv"
) -> Dict[str, Any]:
    vulnerabilities_file = os.path.join(smartbugs_dir, "vulnerabilities.json")
    if not os.path.exists(vulnerabilities_file):
        raise FileNotFoundError(f"Missing {vulnerabilities_file}. Ensure smartbugs-curated is cloned.")

    with open(vulnerabilities_file, "r", encoding="utf-8") as f:
        vuln_data = json.load(f)

    gold_records = []
    category_counts = {}
    pragma_versions = {}

    for item in vuln_data:
        contract_path = os.path.join(smartbugs_dir, item["path"])
        if not os.path.exists(contract_path):
            continue

        with open(contract_path, "r", encoding="utf-8", errors="replace") as cf:
            raw_source = cf.read()

        # Clean source using line-preserving state machine
        clean_code = clean_source(raw_source)

        # Safety verification: no leakage of annotations
        for forbidden in ["@vulnerable_at_lines", "<yes>", "<report>"]:
            if forbidden in clean_code:
                # Extra pass to strip any residual inline tags
                clean_code = clean_code.replace(forbidden, " " * len(forbidden))

        # Extract pragma version if present
        pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', raw_source)
        pragma_str = pragma_match.group(1).strip() if pragma_match else item.get("pragma", "unknown")
        pragma_versions[pragma_str] = pragma_versions.get(pragma_str, 0) + 1

        # Combine vulnerable lines and primary category
        all_vuln_lines = set()
        categories = set()
        for v in item.get("vulnerabilities", []):
            cat = v.get("category", "other")
            categories.add(cat)
            lines = v.get("lines", [])
            for ln in lines:
                all_vuln_lines.add(int(ln))

        primary_cat = sorted(list(categories))[0] if categories else "clean"
        sorted_lines = sorted(list(all_vuln_lines))

        record = {
            "id": item["name"],
            "path": item["path"],
            "category": primary_cat,
            "categories": sorted(list(categories)),
            "pragma": pragma_str,
            "source_raw_lines": len(raw_source.splitlines()),
            "source_clean_lines": len(clean_code.splitlines()),
            "source_clean": clean_code,
            "vulnerable_lines": sorted_lines
        }
        gold_records.append(record)

        # Track category statistics
        for cat in (categories if categories else ["clean"]):
            if cat not in category_counts:
                category_counts[cat] = {
                    "contracts": 0,
                    "total_vuln_lines": 0
                }
            category_counts[cat]["contracts"] += 1
            category_counts[cat]["total_vuln_lines"] += len(sorted_lines)

    # Save JSON fallback format
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(gold_records, jf, indent=2)

    # Save CSV counts report
    os.makedirs(os.path.dirname(counts_csv), exist_ok=True)
    with open(counts_csv, "w", encoding="utf-8") as cf:
        cf.write("category,contracts,total_vulnerable_lines\n")
        for cat, stats in sorted(category_counts.items()):
            cf.write(f"{cat},{stats['contracts']},{stats['total_vuln_lines']}\n")

    # Try writing Parquet if pyarrow/pandas is available
    try:
        import pandas as pd
        df = pd.DataFrame(gold_records)
        df.to_parquet(output_parquet, index=False)
        parquet_saved = True
    except ImportError:
        parquet_saved = False

    return {
        "total_contracts": len(gold_records),
        "categories": category_counts,
        "pragma_versions": pragma_versions,
        "parquet_saved": parquet_saved
    }


if __name__ == "__main__":
    res = build_smartbugs_gold()
    print(f"SmartBugs Gold Set Built: {res['total_contracts']} contracts processed.")
    print(f"Parquet saved: {res['parquet_saved']}")
