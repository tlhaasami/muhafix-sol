# Muhafiz.sol: Report Chapters 1 to 3
**Author:** Talha Sami (24I-3118), Phase 1 Owner  
**Course:** AI2002 — Artificial Intelligence (Fall 2026)  
**Project:** Muhafiz.sol (Smart Contract Vulnerability Detector)  

---

## Chapter 1: Problem Formulation and Justification

### 1.1 The Context of Smart Contract Security
Decentralized applications on the Ethereum blockchain entrust billions of dollars in economic value to smart contracts executed by the Ethereum Virtual Machine (EVM). Two intrinsic characteristics make Solidity smart contract development unusually high-stakes:
1. **Direct Value Custody:** Smart contracts manage financial custody directly through native ether and token balances.
2. **Immutability:** Once bytecode is deployed to a blockchain address, runtime code cannot be unilaterally modified or patched through conventional hotfixes.

Flaws in smart contracts (such as reentrancy, access control bypasses, and unhandled low-level calls) historically led to catastrophic exploits (e.g., The DAO hack, Parity multi-sig freeze).

### 1.2 Limitations of Existing Tools
Traditional static analysis tools such as Slither and Mythril provide valuable formal checks but present operational bottlenecks:
* They require precise compiler versions and compilation environments, failing on uncompilable or fragmented snippets.
* They incur high runtime latency when scanning extensive repositories.
* They lack probabilistic confidence scores and contextual similarity retrieval for developer education.

### 1.3 Project Justification & Problem Statement
**Muhafiz.sol** addresses this gap by formulating vulnerability detection as an interpretable multi-label text classification problem. Operating directly on raw Solidity source code, Muhafiz.sol provides instantaneous vulnerability triage, risk scoring, and line-level token explanations without requiring compilation. It is positioned as an accessible first-pass security triage aid for student developers and emerging Web3 teams.

---

## Chapter 2: Methodology and Architectural Overview

### 2.1 Multi-Label Task Formulation
Given an uncompiled Solidity source document $S$, the system outputs:
1. A calibrated probability vector $\hat{\mathbf{y}} = [p_1, p_2, \dots, p_K] \in [0, 1]^K$ across $K = 7$ target vulnerability classes.
2. A binary prediction vector based on class-specific operational thresholds $\tau_k$.
3. An aggregate contract risk classification (Low, Medium, High).
4. Line-level importance attribution indicating the lines most responsible for each vulnerability flag.

### 2.2 Dual-Dataset Evaluation Strategy
To prevent self-reinforcing bias and data leakage:
* **Training Corpus:** We train baseline and ML classifiers on the *Slither Audited Smart Contracts* dataset.
* **Independent Gold Benchmark:** We evaluate models against *SmartBugs Curated*, a human-audited benchmark with line-by-line vulnerability annotations.

This separation ensures that performance metrics genuinely reflect semantic generalizability rather than automated label reproduction.

---

## Chapter 3: Dataset Description, Preprocessing, and Deduplication

### 3.1 Data Acquisition
The pipeline ingests:
* 14,135 contracts from the verified Slither collection on Hugging Face.
* 143 human-curated contracts from SmartBugs covering 9 primary vulnerability categories.

### 3.2 Preprocessing and Comment Invariance
Downstream explainability requires line-level mapping between original source code and model feature weights. Conventional code preprocessors strip comments by deleting characters, which destroys line numbers.
Muhafiz.sol implements a custom single-pass state machine (`clean_source` in `src/preprocess.py`) that:
* Replaces comment characters with whitespace while preserving newline characters (`\n`, `\r\n`).
* Masks string literals as `<STR>` and numbers/addresses as `<NUM>`.
* Protects security-critical tokens (`msg.sender`, `tx.origin`, `selfdestruct`, `delegatecall`, `call`, `revert`).
* Ensures that for any contract, the line count of the cleaned text matches the raw source exactly: $\text{Lines}(S_{\text{clean}}) = \text{Lines}(S_{\text{raw}})$.

### 3.3 De-duplication and Group-Aware Splits
Smart contract repositories exhibit extreme duplication rates due to cloned ERC-20 and OpenZeppelin contracts. A naive random split would allow near-identical clones to populate both the training and test sets, severely compromising evaluation honesty.

Our deduplication pipeline executes:
1. **Exact Deduplication:** SHA-256 hashing of normalized token sequences.
2. **Near-Duplicate Clustering:** 5-gram token shingling, 128-permutation MinHash calculation, and Locality-Sensitive Hashing (LSH) with Union-Find connected components ($J \ge 0.80$).
3. **Group-Aware Splitting:** Assigning entire connected components exclusively into Train (70%), Validation (15%), or Test (15%) splits, guaranteeing zero clone leakage.
