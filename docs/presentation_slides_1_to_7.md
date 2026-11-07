# Muhafiz.sol: Presentation Slide Deck (Slides 1 to 7)
**Presenter / Author:** Talha Sami (24I-3118), Phase 1 Lead  
**Topic:** Problem, Architecture, Data Foundation & Preprocessing Pipeline  

---

### Slide 1: Title & Executive Overview
* **Title:** Muhafiz.sol: Machine Learning Smart Contract Vulnerability Detector
* **Subtitle:** AI2002 Semester Project — Group 39 (Fall 2026)
* **Team:** Talha Sami (Phase 1 Lead), Sohaib (Phase 2 Lead), Shaiman (Phase 3 Lead)
* **Core Value Proposition:** Fast, compiler-free, interpretable security triage for Ethereum smart contracts with line-level importance highlighting.

---

### Slide 2: The High-Stakes Problem in Web3
* **Direct Value Custody:** Smart contracts manage financial assets on-chain ($ billions total value locked).
* **Code Immutability:** Once deployed to Ethereum, contracts cannot be patched; single flaws lead to irrevocable financial loss.
* **The Developer Dilemma:** Full security audits take weeks and cost thousands of dollars; static tools require complex local compiler configurations.
* **Our Mission:** Create a lightweight, immediate vulnerability triage tool for student and emerging developers.

---

### Slide 3: Architectural Blueprint & Interface Contract
* **Input:** Raw Solidity source text (no compiler required).
* **Pipeline:**
  1. *Phase 1 (Talha):* Line-preserving cleaning, token normalization, TF-IDF feature extraction.
  2. *Phase 2 (Sohaib):* Multi-label classifiers, calibrated thresholds, and risk scoring.
  3. *Phase 3 (Shaiman):* Line-level token attribution (LIME/attention) & Streamlit web application.
* **Output:** Risk rating (Low/Medium/High), predicted vulnerability probabilities, and highlighted vulnerable code lines.

---

### Slide 4: Honest Dual-Dataset Design
* **Dataset A (Slither Audited Contracts):**
  * ~14K–106K verified contracts from Hugging Face with Slither static analysis labels.
  * Used for training, validation, and feature vocabulary construction.
* **Dataset B (SmartBugs Curated Gold Benchmark):**
  * 143 human-audited contracts with line-by-line ground truth annotations.
  * Used for independent, un-compromised evaluation.
* **Why Two Datasets?** Prevents the model from merely memorizing one static analysis tool's quirks.

---

### Slide 5: Scientific Data Preprocessing
* **The Line Invariance Challenge:** Deleting comments destroys original line numbers, making explainability and line highlighting impossible downstream.
* **Our State-Machine Solution (`src/preprocess.py`):**
  * Comments are converted to equivalent whitespace; newline characters (`\n`, `\r\n`) are strictly preserved.
  * Line count of cleaned code is **100% identical** to raw code ($L_{\text{raw}} = L_{\text{clean}}$).
* **Domain Normalization:** Literal strings $\to `<STR>`, numbers $\to `<NUM>`, while strictly preserving security-critical primitives (`msg.sender`, `tx.origin`, `delegatecall`, `selfdestruct`).

---

### Slide 6: Combating Clone Leakage (MinHash LSH Deduplication)
* **The Blockchain Clone Epidemic:** Over 30% of contracts on Ethereum are clones or slight forks of token templates.
* **The Flaw of Random Splits:** Clones sitting in both training and test sets falsely inflate accuracy.
* **Our Group-Aware Deduplication (`src/splits.py`):**
  1. *Exact:* SHA-256 token hashing.
  2. *Near-Duplicate:* 5-gram shingles + MinHash LSH ($J \ge 0.80$) with Union-Find clustering.
  3. *Strict Group Splitting:* Complete clone groups assigned exclusively to Train (70%), Val (15%), or Test (15%). Zero clone leakage.

---

### Slide 7: Phase 1 Quality Gate & Acceptance Verifications
* **Acceptance Test A1 (Zero Annotation Leakage):** Passed. Zero inline annotation markers in cleaned gold contracts.
* **Acceptance Test A2 (Zero Group Leakage):** Passed. Null intersection between train, val, and test group IDs.
* **Acceptance Test A3 (Line Preservation):** Passed. Verified across 15 automated test cases in `test_preprocess.py`.
* **Acceptance Test A4 (Deterministic Rebuilds):** Passed. Reproducible splits with seed 42.
* **Handoff Ready:** Cleaned data packs, feature pipelines, and label maps delivered to Phase 2 (Sohaib) and Phase 3 (Shaiman).
