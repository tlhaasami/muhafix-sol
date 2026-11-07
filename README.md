# Muhafiz.sol 🛡️
### Ethereum Smart Contract Vulnerability Detector
**AI2002: Artificial Intelligence — Semester Project, Fall 2026 (Group 39)**

---

## 👥 Group Members
* **Talha Sami (24I-3118)** — Phase 1 Owner: Data Foundation & Feature Pipeline
* **Sohaib (24I-3108)** — Phase 2 Owner: Modelling, Tuning & Evaluation
* **Shaiman (24I-3074)** — Phase 3 Owner: Explainability, Streamlit App & Delivery

---

## 🚀 Overview
**Muhafiz.sol** (*Muhafiz* means guardian in Urdu; `.sol` is the Solidity file extension) is a machine-learning vulnerability triage tool for Ethereum smart contracts. It scans raw Solidity contracts to predict security vulnerability likelihoods, an overall contract risk score, and highlighted vulnerable source lines.

Built on:
1. **Training Benchmark:** Slither Audited Smart Contracts (Hugging Face)
2. **Independent Gold Evaluation:** SmartBugs Curated (GitHub)

---

## 📁 Repository Structure
```
muhafiz-sol/
├── config.yaml               # Centralized configuration, seeds, and paths
├── requirements.txt          # Python dependencies
├── PHASE_1_EXECUTION_TRACKER.md # Phase 1 real-time progress tracker
├── README.md
├── data/
│   ├── raw/                  # Downloaded raw datasets
│   ├── processed/            # De-duplicated, group-aware train/val/test splits
│   ├── gold/                 # Cleaned SmartBugs benchmark (line-level ground truth)
│   └── features/             # Serialized sparse TF-IDF matrices
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Line-preserving cleaner, normalizer, tokenizer
│   ├── features.py           # TF-IDF pipeline and SVD representations
│   ├── splits.py             # MinHash LSH deduplication & group split builder
│   └── build_data.py         # CLI runner for one-command end-to-end data build
├── models/                   # Saved vectorizers, label maps, and trained weights
├── notebooks/                # EDA and data audit notebooks
├── tests/                    # Unit and integration test suites (pytest)
├── docs/                     # Data card, label decisions, dedup reports
└── reports/
    ├── figures/              # Visualizations and EDA charts
    └── tables/               # Distribution summaries and metrics
```

---

## 🛠️ Quickstart

### 1. Environment Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest
```

### 3. Build Data Pipeline (Phase 1)
```bash
python -m src.build_data
```
