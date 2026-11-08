"""
Build the single submission zip (blueprint P3.6) and check that it runs from a clean folder.

    python tools/make_submission.py            # writes dist/Muhafiz_sol_Group39.zip
    python tools/make_submission.py --verify   # also unzips to a temp folder and scans a sample there

Contents: every file tracked by git (source, tests, notebooks, docs, figures, tables, the frozen model
files, sample contracts, README, requirements) plus the final report and slides. The large raw and
intermediate data are not included; `python -m src.build_data` rebuilds them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Muhafiz_sol_Group39"
EXTRA = ["reports/Muhafiz_sol_Final_Report.docx", "reports/Muhafiz_sol_Slides.pptx"]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and (ROOT / f).is_file()]


def build() -> Path:
    files = sorted(set(tracked_files()) | {f for f in EXTRA if (ROOT / f).is_file()})
    missing = [f for f in EXTRA if not (ROOT / f).is_file()]
    if missing:
        print("warning: not found (run python -m src.make_report first):", ", ".join(missing))
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"{NAME}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(ROOT / f, f"{NAME}/{f}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    return out


def verify(zip_path: Path) -> None:
    """Unzip into an empty temp folder and scan two sample contracts with the frozen model there."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        root = Path(tmp) / NAME
        code = ("from pathlib import Path\n"
                "from src.predict import Predictor\n"
                "p = Predictor()\n"
                "for f in ['01_clean_token.sol', '02_reentrancy_bank.sol']:\n"
                "    r = p.predict((Path('app/sample_contracts') / f).read_text(encoding='utf-8'))\n"
                "    print(f, r['risk_level'], {k: round(v, 2) for k, v in r['probabilities'].items()})\n")
        env = {**os.environ, "PYTHONPATH": str(root)}
        res = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, env=env)
        print(res.stdout.strip() or res.stderr.strip())
        if res.returncode != 0:
            raise SystemExit("verification failed")
        # the test suite must pass in the unzipped copy too (tests that need the large data skip themselves)
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root,
                               capture_output=True, text=True, env=env)
        print(tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else tests.stderr.strip())
        if tests.returncode != 0:
            print(tests.stdout[-3000:])
            raise SystemExit("verification failed: tests fail in the unzipped copy")
        print("verification passed: the unzipped copy loads the model, scans contracts and passes its tests")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    z = build()
    if args.verify:
        verify(z)
