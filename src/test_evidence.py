"""
Run the test suite and record the evidence (blueprint P3.4):
docs/test_evidence.md (human readable) and docs/test_evidence.json (used by the report).

    python -m src.test_evidence
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from src.config import ROOT, path

ACCEPTANCE = [
    ("A1", "No annotation leakage", "test_data_pack::test_a1_no_annotation_text_reaches_any_model_input, test_gold::test_a1_no_annotation_leakage"),
    ("A2", "No group leakage", "test_data_pack::test_a2_no_group_in_two_splits"),
    ("A3", "Line preservation (1,000 random contracts)", "test_data_pack::test_a3_line_count_preserved_on_1000_dataset_a_contracts, test_preprocess::test_a3_line_count_invariant_on_random_inputs"),
    ("A4", "Determinism (identical split hashes)", "test_data_pack::test_a4_split_is_reproducible"),
    ("A5", "Class coverage in every split", "test_data_pack::test_a5_every_class_has_positives_in_every_split"),
    ("A6", "One-command rebuild", "test_data_pack::test_a6_one_command_build_entry_point_exists (+ full run of python -m src.build_data)"),
    ("B1", "Metric correctness on hand-made arrays", "test_evaluate (9 tests)"),
    ("B2", "No test-set use in selection", "src/train.py refuses test/gold; run_experiments checks tag model-v1 (test_freeze::*)"),
    ("B3", "Reproducibility", "fixed seeds; split hashes; test_freeze::test_logreg_retrain_reproduces_frozen_validation_score"),
    ("B4", "Interface schema", "test_integration::test_six_real_contracts_give_sensible_output"),
    ("B5", "Seed stability", "reports/tables/e1_mlp_seeds_test.csv (spread reported)"),
    ("B6", "Honest baselines", "prior and logistic regression present in every result table (test_freeze::test_baselines_in_every_table)"),
    ("C1", "One-command run", "streamlit run app/app.py on the development laptop; other two laptops pending (team)"),
    ("C2", "Ten fixed inputs", "test_integration (six contracts + four edge cases), table below"),
    ("C3", "App output equals predict()", "test_integration::test_app_shows_exactly_the_predict_output"),
    ("C4", "Explanation quality", "E3b hit-rate with intervals (reports/tables/e3b_hit_rate.csv)"),
    ("C5", "Latency", "test_integration::test_latency_is_a_few_seconds; reports/tables/e5_latency.csv"),
    ("C6", "Package runs on a clean machine", "fresh venv install from requirements.txt + pytest (pending: zip on a second machine)"),
]


def run_pytest() -> dict:
    xml = path("reports_dir") / "pytest_results.xml"
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}", "-p", "no:cacheprovider"],
                   cwd=ROOT, check=False)
    secs = time.time() - t0
    root = ET.parse(xml).getroot()
    files = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "names": []})
    for tc in root.iter("testcase"):
        mod = tc.get("classname", "").split(".")[-1]
        status = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        elif tc.find("skipped") is not None:
            status = "skipped"
        files[mod][status] += 1
        files[mod]["names"].append((tc.get("name"), status))
    xml.unlink(missing_ok=True)
    total = {k: sum(v[k] for v in files.values()) for k in ("passed", "failed", "skipped")}
    return {"seconds": secs, "files": files, **total, "total": sum(total.values())}


def ten_inputs() -> list:
    from src.predict import InputError, Predictor

    p = Predictor()
    rows = []
    expect = {
        "01_clean_token.sol": "low risk or few flags (clean token)",
        "02_reentrancy_bank.sol": "reentrancy flagged",
        "03_unchecked_send.sol": "unchecked-calls flagged",
        "04_weak_randomness_lottery.sol": "bad-randomness flagged",
        "05_tricky_safe_with_keywords.sol": "lower reentrancy score than 02 (correct use of call)",
        "06_tricky_renamed_reentrancy.sol": "reentrancy flagged despite odd names",
    }
    for f in sorted((ROOT / "app" / "sample_contracts").glob("*.sol")):
        r = p.predict(f.read_text(encoding="utf-8"))
        flagged = [c for c, v in r["flags"].items() if v]
        top = max(r["probabilities"], key=r["probabilities"].get)
        rows.append({"input": f.name, "expected": expect[f.name], "risk": r["risk_level"],
                     "flagged": ", ".join(flagged) or "-", "highest": f"{top} {r['probabilities'][top]:.2f}"})
    for name, text, exp in [("empty", "", "clear message"), ("not Solidity", "Hello, this is a shopping list. " * 10,
                                                              "warning, no crash"),
                            ("oversized", "contract A {}" * 200000, "clear message"), ("very short", "x;", "clear message")]:
        try:
            r = p.predict(text)
            obs = ("warning: " + r["warnings"][0][:60]) if r["warnings"] else f"scored ({r['risk_level']})"
            rows.append({"input": name, "expected": exp, "risk": r["risk_level"], "flagged": "-", "highest": obs})
        except InputError as exc:
            rows.append({"input": name, "expected": exp, "risk": "-", "flagged": "-", "highest": f"message: {str(exc)[:60]}"})
    return rows


def main() -> None:
    res = run_pytest()
    names = {f"{mod}::{n}": s for mod, v in res["files"].items() for n, s in v["names"]}
    acc = []
    for tid, title, evidence in ACCEPTANCE:
        linked = [k for k in names if any(part.strip().split(" ")[0] in k for part in evidence.split(","))]
        statuses = {names[k] for k in linked}
        status = ("passed" if statuses == {"passed"} else "FAILED" if "failed" in statuses else
                  "skipped" if statuses == {"skipped"} else "see evidence")
        if tid in ("C1", "C6"):
            status = "partly (team)"
        acc.append({"test": tid, "check": title, "status": status, "evidence": evidence})
    try:
        inputs = ten_inputs()
    except FileNotFoundError:
        inputs = []
    out = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "seconds": res["seconds"], "total": res["total"],
        "passed": res["passed"], "failed": res["failed"], "skipped": res["skipped"],
        "files": [{"file": f"tests/{m}.py", "passed": v["passed"], "failed": v["failed"],
                   "covers": ", ".join(n for n, _ in v["names"][:4]) + (" ..." if len(v["names"]) > 4 else "")}
                  for m, v in sorted(res["files"].items())],
        "acceptance": acc, "ten_inputs": inputs,
    }
    (path("docs_dir") / "test_evidence.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Test evidence\n\nGenerated by `python -m src.test_evidence` on {out['date']}.\n",
          f"**{out['passed']} passed, {out['failed']} failed, {out['skipped']} skipped** of {out['total']} tests "
          f"({out['seconds']:.0f} s).\n", "## Acceptance tests\n", "| Test | Check | Status | Evidence |", "|---|---|---|---|"]
    md += [f"| {a['test']} | {a['check']} | {a['status']} | {a['evidence']} |" for a in acc]
    md += ["\n## Ten fixed inputs (C2)\n", "| Input | Expected | Risk | Flagged | Observed |", "|---|---|---|---|---|"]
    md += [f"| {r['input']} | {r['expected']} | {r['risk']} | {r['flagged']} | {r['highest']} |" for r in inputs]
    md += ["\n## Test files\n", "| File | Passed | Failed |", "|---|---|---|"]
    md += [f"| {f['file']} | {f['passed']} | {f['failed']} |" for f in out["files"]]
    md += ["\n## Manual checks still owned by the team\n",
           "* C1: `streamlit run app/app.py` on the other two laptops.",
           "* C6: unzip the submission on a clean machine and run it.",
           "* Mock viva and two demo rehearsals (3 and 5 December)."]
    (path("docs_dir") / "test_evidence.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"tests: {out['passed']} passed, {out['failed']} failed, {out['skipped']} skipped")


if __name__ == "__main__":
    main()
