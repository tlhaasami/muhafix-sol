# Dataset Card: Muhafiz.sol Benchmark & Gold Datasets
**Project:** Muhafiz.sol — Smart Contract Vulnerability Detector (AI2002)  
**Author:** Talha Sami (24I-3118), Phase 1 Owner  
**Date:** 27 October 2026  
**Version:** 1.0.0  

---

## 1. Dataset Overview & Provenance

Muhafiz.sol utilizes a dual-dataset design to guarantee objective, leak-free evaluation:

| Dataset | Role | Upstream Repository / Source | Size / Instances | Annotation Type |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset A (Slither Audited)** | Training & Validation | Hugging Face: `mwritescode/slither-audited-smart-contracts` | ~14K–106K contracts | Automated labels from Slither static analyzer |
| **Dataset B (SmartBugs Curated)** | Independent Gold Test | GitHub: `smartbugs/smartbugs-curated` | 143 contracts | Hand-annotated, line-level human security audit ground truth |

---

## 2. Licensing and Ethical Attribution
* **Slither Audited Smart Contracts:** MIT License (Martina Rossini, 2022). Verified open-source contracts sourced from Etherscan.io.
* **SmartBugs Curated Dataset:** MIT License (Ferreira et al., ICSE 2020). Open-source benchmark for security tool evaluation.
* **Intended Use:** Fast first-pass triage aid for Solidity developers and Web3 development teams. Not a certified replacement for formal verification or professional audits.

---

## 3. Preprocessing & Normalization Pipeline

To eliminate spurious artifacts while maintaining complete interpretability:
1. **Comment Stripping with Invariant Line Counts (`clean_source`):**
   * Single-pass state machine removes single-line (`//`) and multi-line (`/* ... */`) comments by substituting whitespace.
   * Total line count and line break positions are strictly preserved ($L_{\text{raw}} = L_{\text{clean}}$), ensuring downstream line-level explanations remain perfectly aligned.
2. **Token Normalization (`normalise`):**
   * String literals replaced with `<STR>`.
   * Numerical and hexadecimal addresses replaced with `<NUM>`.
   * Security-sensitive primitives and members (`tx.origin`, `msg.sender`, `msg.value`, `delegatecall`, `selfdestruct`, `block.timestamp`, etc.) are protected from obfuscation.
3. **Tokenization (`tokenize`):**
   * Regular expression parsing that isolates compound Solidity built-in expressions as atomic tokens.

---

## 4. Deduplication & Group-Aware Splits

Smart contract ecosystems suffer heavily from code cloning (forked tokens, boilerplates). A standard random split leaks clone variants across splits, artificially inflating evaluation metrics.

* **Exact Deduplication:** SHA-256 hash of normalized token sequences.
* **Near-Duplicate Clustering:** 
  * 5-gram token shingles.
  * 128-permutation MinHash signatures via Locality-Sensitive Hashing (LSH).
  * Connected components resolved via Union-Find with Jaccard similarity threshold $\ge 0.80$.
* **Group-Aware Splitting:**
  * Entire clone groups are placed exclusively into either `train` (70%), `validation` (15%), or `test` (15%).
  * **Zero Group Leakage:** Train, validation, and test group sets have null intersection.
  * An un-grouped random split of identical size is maintained to quantify clone-leakage inflation (Experiment E2).

---

## 5. Target Vulnerability Classes

| Target ID | Vulnerability Class | Slither Detectors Included | SmartBugs Category |
| :---: | :--- | :--- | :--- |
| **0** | `access-control` | `suicidal`, `arbitrary-send`, `controlled-delegatecall`, `tx-origin` | `access_control` |
| **1** | `arithmetic` | `divide-before-multiply` | `arithmetic` |
| **2** | `reentrancy` | `reentrancy-eth`, `reentrancy-no-eth` | `reentrancy` |
| **3** | `unchecked-calls` | `unused-return`, `unchecked-lowlevel`, `unchecked-transfer` | `unchecked_low_level_calls` |
| **4** | `locked-ether` | `locked-ether`, `unchecked-send` | `denial_of_service` |
| **5** | `bad-randomness` | `weak-prng` | `bad_randomness`, `time_manipulation` |
| **6** | `other` | Miscellaneous low-frequency warnings | Miscellaneous |

*Note on Safe Contracts:* Label 4 in Slither indicates a contract with zero detected flaws. Contracts are classified as safe when all vulnerability probabilities fall below decision thresholds.

---

## 6. Dataset Limitations & Threats to Validity
1. **Tool-Generated Labels:** Training labels reflect Slither's static analysis output, which can include false positives or false negatives.
2. **Compiler Invariance:** Because Muhafiz.sol processes uncompiled source text, it does not execute AST construction or bytecode analysis, prioritizing triage speed over exhaustive semantic proof.
3. **Benchmark Disparity:** SmartBugs contains older Solidity versions (0.4.x–0.5.x), whereas live contracts often use 0.8.x with built-in overflow protection.
