"""
Muhafiz.sol - SmartBugs Curated gold set (blueprint P1.3).

Dataset B is used for *testing only*. Every contract is cleaned with the
line-preserving routine so the ``@vulnerable_at_lines`` header and the inline
``// <yes> <report> CATEGORY`` markers disappear while the annotated line
numbers stay valid. Unlike the earlier prototype, leftover markers are *not*
patched over: if any survive cleaning the build fails, because that would mean
the answers could leak into the model input (acceptance test A1).

Outputs
  data/gold/smartbugs_gold.parquet   id, category, source_clean, vulnerable_lines, ...
  reports/tables/smartbugs_counts.csv
  reports/tables/smartbugs_pragmas.csv

Run:  python -m src.build_gold
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import load_config, path
from src.preprocess import clean_source, count_lines, split_lines

SMARTBUGS_COMMIT = "230e649123477eff332742a59a1c7cc6dc286cab"  # pinned (16 Jun 2024)
LEAK_MARKERS = ("@vulnerable_at_lines", "<yes>", "<report>", "@source", "@author")
_PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);")


def ensure_smartbugs(dest: Path | None = None) -> Path:
    """Clone SmartBugs Curated at the pinned commit if it is not present."""
    dest = dest or path("smartbugs_dir")
    if (dest / "vulnerabilities.json").exists():
        return dest
    url = load_config()["dataset"]["smartbugs_git_repo"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", SMARTBUGS_COMMIT], check=True)
    return dest


def _marker_consistency(raw_lines: List[str], vuln_lines: List[int]) -> int:
    """Count annotated lines whose ``<yes> <report>`` marker is on the line above
    or on the same line (the SmartBugs convention). Used only as a sanity check."""
    ok = 0
    for ln in vuln_lines:
        window = raw_lines[max(0, ln - 2):ln]  # line above and the line itself
        if any("<yes>" in w and "<report>" in w for w in window):
            ok += 1
    return ok


def build_smartbugs_gold(smartbugs_dir: Path | None = None) -> Dict[str, Any]:
    smartbugs_dir = ensure_smartbugs(smartbugs_dir)
    vulns = pd.read_json(smartbugs_dir / "vulnerabilities.json")
    icse69 = {
        line.strip().lstrip("./")
        for line in (smartbugs_dir / "ICSE2020_curated_69.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().endswith(".sol")
    }

    records: List[Dict[str, Any]] = []
    counts: Dict[str, Counter] = defaultdict(Counter)
    pragmas: Counter = Counter()
    marker_ok = marker_total = 0

    for item in vulns.to_dict("records"):
        raw = (smartbugs_dir / item["path"]).read_text(encoding="utf-8", errors="replace")
        clean = clean_source(raw)
        leaked = [m for m in LEAK_MARKERS if m in clean]
        if leaked:
            raise AssertionError(f"annotation marker(s) {leaked} survived cleaning in {item['path']}")
        if count_lines(clean) != count_lines(raw):
            raise AssertionError(f"line count changed while cleaning {item['path']}")

        by_cat: Dict[str, List[int]] = defaultdict(list)
        for v in item["vulnerabilities"]:
            by_cat[v["category"]].extend(int(x) for x in v["lines"])
        categories = sorted(by_cat)
        all_lines = sorted({ln for lines in by_cat.values() for ln in lines})
        n_lines = count_lines(raw)
        if any(ln < 1 or ln > n_lines for ln in all_lines):
            raise AssertionError(f"annotated line outside file in {item['path']}")

        raw_lines = split_lines(raw)
        marker_ok += _marker_consistency(raw_lines, all_lines)
        marker_total += len(all_lines)

        m = _PRAGMA.search(clean)
        pragma = m.group(1).strip() if m else "none"
        pragmas[pragma] += 1
        records.append({
            "id": item["path"].replace("dataset/", "", 1),
            "name": item["name"],
            "path": item["path"],
            "category": categories[0],          # SmartBugs has one category per file
            "categories": categories,
            "vulnerable_lines": all_lines,
            "lines_by_category": {c: sorted(set(v)) for c, v in by_cat.items()},
            "n_vulnerabilities": len(item["vulnerabilities"]),
            "pragma": pragma,
            "compiled_version": str(item.get("pragma", "")),
            "n_lines": n_lines,
            "icse2020_69": item["path"] in icse69,
            "source_clean": clean,
        })
        for c in categories:
            counts[c]["contracts"] += 1
            counts[c]["annotated_vulnerabilities"] += sum(
                1 for v in item["vulnerabilities"] if v["category"] == c)
            counts[c]["annotated_lines"] += len(set(by_cat[c]))

    gold = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
    out = path("gold_dir") / "smartbugs_gold.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(out, index=False)

    tables = path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    count_df = pd.DataFrame([{"category": c, **v} for c, v in sorted(counts.items())])
    count_df.loc[len(count_df)] = {"category": "TOTAL", **count_df.drop(columns="category").sum().to_dict()}
    count_df.to_csv(tables / "smartbugs_counts.csv", index=False)
    pd.DataFrame(sorted(pragmas.items(), key=lambda x: -x[1]), columns=["pragma", "contracts"]).to_csv(
        tables / "smartbugs_pragmas.csv", index=False)

    return {
        "contracts": len(gold),
        "annotated_vulnerabilities": int(gold["n_vulnerabilities"].sum()),
        "annotated_lines": int(gold["vulnerable_lines"].map(len).sum()),
        "marker_consistency": f"{marker_ok}/{marker_total}",
        "icse2020_69_found": int(gold["icse2020_69"].sum()),
        "path": str(out),
    }


if __name__ == "__main__":
    for k, v in build_smartbugs_gold().items():
        print(f"{k}: {v}")
