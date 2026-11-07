# Phase 1 Execution Tracker: Data Foundation & Feature Pipeline
**Project:** Muhafiz.sol (Smart Contract Vulnerability Detector) — AI2002 Fall 2026 (Group 39)  
**Phase 1 Owner:** Talha Sami (Roll No: 24I-3118)  
**Teammates:** Sohaib (24I-3108, Phase 2 Owner), Shaiman (24I-3074, Phase 3 Owner)  
**Planned Effort:** 46h Phase 1 + 8h Joint Work = 54h Total  
**Calendar Window:** 26 October – 15 November 2026  

---

## 1. Executive Summary & Responsibility

Phase 1 provides the foundational integrity for the entire Muhafiz.sol project. All subsequent models, metrics, explainability features, and evaluations depend directly on the dataset splits, preprocessing reliability, and leak-prevention mechanisms created in this phase.

* **Primary Objective:** Build an end-to-end reproducible, leak-free data preprocessing, deduplication, and feature extraction pipeline.
* **Key Handoffs:**
  * **H1 (31 Oct):** Sample data pack for early baseline modeling.
  * **H2 (11 Nov, Gate G1):** Full cleaned splits, TF-IDF vectorizer, SmartBugs gold set, label mapping, and data card.

---

## 2. Work Package Progress Status

| Work Package ID | Title | Estimated Hours | Target Window | Status | Output Files / Artifacts |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **P1.1** | Repository & Environment | 4 h | 26–27 Oct | **COMPLETED** | `requirements.txt`, `config.yaml`, `.gitignore`, `README.md`, folder skeleton, `tests/test_environment.py` |
| **P1.2** | Audit Dataset A (Slither) | 8 h | 27–30 Oct | **COMPLETED** | `docs/label_decision.md`, `data/label_mappings.json` (Label 4 = `'safe'` formally resolved) |
| **P1.3** | Build SmartBugs Gold Set | 6 h | 2–6 Nov | **COMPLETED** | `data/raw/smartbugs-curated/`, `data/gold/smartbugs_gold.json`, `reports/tables/smartbugs_counts.csv`, `tests/test_gold.py` |
| **P1.4** | Preprocessing Module & Unit Tests | 8 h | 31 Oct–5 Nov | **COMPLETED** | `src/preprocess.py`, `tests/test_preprocess.py` (15/15 unit tests passing) |
| **P1.5** | De-duplication & Group Splits | 8 h | 2–11 Nov | **COMPLETED** | `src/splits.py`, `data/processed/` (`train.json`, `val.json`, `test.json`, `random_test.json`), `docs/dedup_report.md` |
| **P1.6** | Class Mapping & Target Set | 4 h | 5–10 Nov | **COMPLETED** | `models/label_map.json`, `data/label_map.json` |
| **P1.7** | TF-IDF Feature Pipeline | 4 h | 9–11 Nov | **COMPLETED** | `src/features.py`, `models/vectorizer.joblib` |
| **P1.8** | Data Card, EDA & Report Sections | 4 h | 9–15 Nov | **COMPLETED** | `docs/data_card.md`, `docs/report_chapters_1_to_3.md`, `docs/presentation_slides_1_to_7.md`, `src/build_data.py`, `reports/tables/label_distribution.csv` |

---

## 3. Phase 1 Quality Gate & Acceptance Tests Checklist

### Gate G1 (Target: 11 November 2026)
* **Owner:** Talha Sami
* **Reviewer:** Shaiman
* **Status:** **READY FOR REVIEW / CRITERIA MET**

| ID | Acceptance Test | Success Criteria | Status |
| :---: | :--- | :--- | :---: |
| **A1** | No Annotation Leakage | Model inputs & gold evaluation text contain zero markers (`@vulnerable_at_lines`, `<yes>`, `<report>`) | ✅ **PASSED** (Validated in `test_gold.py`) |
| **A2** | No Group / Clone Leakage | Empty intersection between clone group IDs in train, val, and test splits | ✅ **PASSED** (Validated in `test_splits.py`) |
| **A3** | Line Count Preservation | Exact line count match before and after comment stripping on contracts | ✅ **PASSED** (Validated in `test_preprocess.py`) |
| **A4** | Deterministic Rebuilds | Identical SHA-256 hashes for data splits when regenerated with seed `42` | ✅ **PASSED** (Validated in `test_splits.py`) |
| **A5** | Multi-Label Class Coverage | Every target vulnerability class has positive instances across all splits | ✅ **PASSED** (Validated in `reports/tables/label_distribution.csv`) |
| **A6** | One-Command Reproduction | `python -m src.build_data` executes cleanly and end-to-end without manual intervention | ✅ **PASSED** (Verified in console) |

---

## 4. Phase Cross-Review Duties

* **Gate G2 (22 Nov) Reviewer:** Talha Sami reviews Phase 2 (Sohaib's model training, metrics, thresholds, predict API, and tests B1–B6).

---

## 5. Live Activity Log

* **2026-10-27:**
  * **[P1.1 COMPLETED] Repository & Environment:**
    * Full folder skeleton, `config.yaml`, `requirements.txt`, `.gitignore`, and `README.md` configured.
    * `tests/test_environment.py` created and passed (3/3).
  * **[P1.2 COMPLETED] Dataset Audit & Label Resolution:**
    * Hugging Face upstream definitions audited.
    * Resolved **Label 4 = `'safe'`** (contracts with zero detected flaws).
    * Created `docs/label_decision.md` explaining mapping logic.
  * **[P1.3 COMPLETED] SmartBugs Gold Set:**
    * Cloned `smartbugs/smartbugs-curated`.
    * Parsed 143 curated contracts with line annotations.
    * Created `src/build_gold.py`, `data/gold/smartbugs_gold.json`, `reports/tables/smartbugs_counts.csv`.
    * Implemented `tests/test_gold.py` (**4/4 passed**, Acceptance Test A1 verified).
  * **[P1.4 COMPLETED] Preprocessing Module:**
    * Implemented `src/preprocess.py` (`clean_source`, `normalise`, `tokenize`, `truncate_tokens`).
    * Implemented `tests/test_preprocess.py` (**15/15 passed**, Acceptance Test A3 verified).
  * **[P1.5 COMPLETED] De-duplication & Group-Aware Splits:**
    * Implemented `src/splits.py` (SHA-256 exact dedup, 5-gram MinHash LSH, Union-Find clustering, 70/15/15 group-aware splits, un-grouped random split).
    * Implemented `tests/test_splits.py` (**3/3 passed**, Acceptance Tests A2 and A4 verified).
    * Generated `docs/dedup_report.md`.
  * **[P1.6 COMPLETED] Class Mapping & Target Set:**
    * Created `models/label_map.json` and `data/label_map.json` establishing the canonical 7 vulnerability classes.
  * **[P1.7 COMPLETED] TF-IDF Feature Pipeline:**
    * Implemented `src/features.py` with n-gram (1 to 3) TF-IDF and 512-dim TruncatedSVD.
  * **[P1.8 COMPLETED] Data Card, Reports & Presentation Slides:**
    * Created `docs/data_card.md`.
    * Authored `docs/report_chapters_1_to_3.md` (Thesis chapters 1–3).
    * Authored `docs/presentation_slides_1_to_7.md` (Slide deck 1–7).
    * Created `src/build_data.py` (One-command pipeline runner, Acceptance Test A6 verified).
  * **Test Suite Verification:** Ran full test suite (`python -m pytest`): **25/25 tests passed (100% pass rate)**.
