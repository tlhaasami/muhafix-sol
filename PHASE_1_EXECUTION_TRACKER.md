# Phase 1 tracker - data foundation and feature pipeline

**Owner:** Talha Sami (24I-3118) · **Reviewer (Gate G1):** Shaiman (24I-3074) · **Window:** 26 Oct - 15 Nov 2026

> Correction (8 Nov 2026): an earlier version of this tracker marked Phase 1 complete while the pipeline had
> only split the 143 SmartBugs *test* contracts and never loaded Dataset A, and recorded teammate sign-offs that
> had not happened. Phase 1 was rebuilt on the real data; the status below reflects what was actually run.

| WP | Output | Status | Evidence |
|---|---|---|---|
| P1.1 Repository and environment | config.yaml, pinned requirements, .gitignore, src/config.py | done | tests/test_environment.py |
| P1.2 Audit Dataset A | label-4 resolution, 0 failed Slither runs, big/small finding, label families | done | docs/label_decision.md, reports/tables/label4_audit.csv, notebooks/01_data_audit.ipynb |
| P1.3 SmartBugs gold set | 143 contracts, 207 vulnerabilities, 222 lines, markers stripped | done | data/gold/smartbugs_gold.parquet, tests/test_gold.py |
| P1.4 Preprocessing | line-preserving cleaner, literal / identifier normalisation, tokenizer, long-file policy | done | src/preprocess.py, tests/test_preprocess.py (31 tests) |
| P1.5 De-duplication and splits | 46 % exact duplicates, 31,186 clone groups, 45,540 / 9,759 / 9,759 split, random split | done | docs/dedup_report.md, data/processed/split_manifest.csv.gz |
| P1.6 Class mapping | 7 classes, SmartBugs mapping (3 confirmed, 1 partial, 6 not evaluated) | done | models/label_map.json |
| P1.7 TF-IDF features | 20,000 1-3-gram vocabulary from train, SVD-512, 4 ms per contract | done | models/vectorizer.joblib, tests/test_features.py |
| P1.8 Data card, EDA, rebuild | data card, EDA figures, one-command rebuild (~50 min) | done | docs/data_card.md, reports/figures, `python -m src.build_data` |

## Acceptance tests (Gate G1)

| Test | Status | Evidence |
|---|---|---|
| A1 No annotation leakage | pass | tests/test_gold.py, tests/test_data_pack.py |
| A2 No group leakage | pass | tests/test_data_pack.py (leakage rate 0.0 %) |
| A3 Line preservation on 1,000 contracts | pass | tests/test_data_pack.py |
| A4 Determinism | pass | split hashes in data/processed/build_summary.json reproduced by tests/test_data_pack.py |
| A5 Class coverage | pass | tests/test_data_pack.py |
| A6 One-command rebuild | pass on the development laptop | `python -m src.build_data` |

## Still to do by the team
* G1 walkthrough with Sohaib and Shaiman; both sign docs/label_decision.md.
* A6 on a second laptop.
