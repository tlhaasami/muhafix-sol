# Muhafiz.sol — Smart Contract Vulnerability Detector

**AI2002 Artificial Intelligence · Semester Project, Fall 2026 · Path A (software) · Group 39**

| Member | Roll no. | Phase |
|---|---|---|
| Talha Sami | 24I-3118 | Phase 1 — data foundation and feature pipeline |
| Sohaib | 24I-3108 | Phase 2 — modelling, tuning and evaluation |
| Shaiman | 24I-3074 | Phase 3 — explainability, Streamlit app and delivery |

*Muhafiz* is Urdu for "guardian"; `.sol` is the Solidity file extension. Muhafiz.sol reads the source code
of an Ethereum smart contract and reports which kinds of security weakness it probably contains, an overall
risk level, the lines most responsible for that judgement, and similar known contracts. All predictions come
from models trained by the group; no rule-based analyser runs at prediction time.

> **Not an audit.** Muhafiz.sol is a first-pass triage aid for students and small teams. It never certifies
> that a contract is safe and is not affiliated with or approved by PVARA or any regulator.

## Quick start (app only, ~2 minutes)

Requires **Python 3.12 or newer** (developed and tested on Python 3.14); the pinned numpy and
scipy versions do not install on older Pythons.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app/app.py
```

The frozen model (`models/`) is committed, so the app runs without downloading or training anything.

## Reproduce everything

```bash
pip install -r requirements-train.txt     # adds PyTorch (CPU) and notebook tools
python -m src.build_data                  # Phase 1: download, clean, de-duplicate, split, TF-IDF (~1 h)
python -m src.train                       # Phase 2: grid search on validation, thresholds, freeze model
git tag model-v1                          # freeze marker: required before any test/SmartBugs evaluation
python -m src.run_experiments             # E1-E4 tables and figures
python -m src.gold_overlap                # which SmartBugs contracts also occur in the training data
python -m src.eval_explain                # E3b line hit-rate, E5 latency / error analysis
python -m src.make_report                 # report (.docx), slides (.pptx), model card refresh
python notebooks/build_notebooks.py       # executes the four notebooks
pytest                                    # unit + integration tests (slow re-training tests are skipped)
RUN_SLOW=1 pytest                         # also re-trains logistic regression and re-runs de-duplication (B3, A4)
```

Every path, seed and hyper-parameter lives in `config.yaml`. Seed 42 everywhere; `python -m src.build_data`
reproduces identical split hashes (`data/processed/build_summary.json`).

## Data

| | Dataset A — training | Dataset B — external test only |
|---|---|---|
| Source | [Slither Audited Smart Contracts](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts) (Rossini, 2022) | [SmartBugs Curated](https://github.com/smartbugs/smartbugs-curated) @ `230e649` |
| Labels | Slither detector output (weak, tool-generated) | Human line-level annotations |
| Use | train / validation / in-distribution test | recall per mapped class, line hit-rate |
| Licence | MIT (repo); contracts keep original licences | Apache-2.0 (repo); contracts keep original licences |

Large files are not committed. `data/processed/split_manifest.csv.gz` (address → group → split) is committed
so the exact splits can be audited. See `docs/data_card.md`, `docs/label_decision.md`, `docs/dedup_report.md`.

## Repository layout

```
config.yaml                 seeds, paths, label rule, features, training, explanation settings
src/
  ingest_slither.py         Dataset A download (column-range reads of the raw parquet shards)
  build_gold.py             SmartBugs gold set (annotations stripped, line numbers preserved)
  preprocess.py             clean_source / normalise / tokenize (line-preserving)
  labels.py                 label families, rare-class rule, SmartBugs mapping
  splits.py                 SHA-256 + MinHash-LSH clone groups, group-aware split, random split
  features.py               TF-IDF (1-3 grams) fitted on train only, SVD-512
  build_data.py             one-command Phase 1 pipeline
  eda.py                    audit tables and EDA figures
  evaluate.py               metrics, bootstrap intervals, thresholds, risk rule, calibration
  models.py                 prior, logistic regression, tree, forest, KNN, MLP (PyTorch)
  train.py                  validation grid, thresholds, calibration, freeze
  predict.py                predict() / similar() API used by the app
  explain.py                two-stage occlusion: risky functions, then risky lines
  run_experiments.py        E1-E4
  eval_explain.py           E3b, E5
  make_report.py            report, slides and model-card generation from the result tables
app/app.py, components.py   Streamlit application; sample_contracts/ holds 6 demo contracts
models/                     frozen vectoriser, models, thresholds, label map, similarity index
notebooks/                  01 data audit, 02 EDA, 03 models, 04 results
tests/                      unit, acceptance (A1-A5, B1-B6, C2-C5; A6/C1/C6 need other laptops) and integration tests
docs/                       data card, model card, label decision, dedup report, test evidence
reports/                    figures, tables, final report and slides
```
