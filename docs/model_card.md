# Model card - Muhafiz.sol model-v1

## Intended use
First-pass triage of Solidity source code by students and small Web3 teams: which weakness families are likely,
an overall risk level, and which lines drive the score. **Not** an audit, never a safety certificate, not a
compliance product, not affiliated with PVARA. Out of scope: bytecode, deployed addresses, non-Ethereum chains,
exploit generation, automatic fixing.

## Model
* **Primary model:** MLP neural network (`mlp`), chosen by validation macro PR-AUC
  (`reports/tables/model_selection_val.csv`). The app also offers logistic regression for comparison.
* Input: TF-IDF of token 1-3-grams (20,000 features) of the cleaned contract.
* Output: one score per class (access-control, arithmetic, bad-randomness, locked-ether, other, reentrancy, unchecked-calls); for the primary model these are not rescaled: the calibration gate (class-averaged ECE > 0.05) was not triggered (ECE 0.047), although several classes are over-confident, so read the percentages as scores.
  Logistic regression's outputs are Platt-scaled.
* MLP architecture: input -> 512 -> 128 -> 7, ReLU, dropout 0.3, sigmoid outputs, binary
  cross-entropy with per-class positive weights (capped at 50), Adam (lr 1e-3), early stopping on validation
  macro PR-AUC (patience 4). Three seeds: validation macro PR-AUC 0.840 ±
  0.004.

## Training data
45,540 training / 9,759 validation / 9,759 test contracts from the Slither
Audited Smart Contracts dataset after exact de-duplication and a clone-group-aware split (see data card).

## Decision thresholds and risk rule
Per-class thresholds maximise F2 on the validation split (recall weighted 4x precision), subject to precision
>= 0.2. Frozen with the model under git tag `model-v1` **before** any test or SmartBugs
evaluation.

* **low** - no class reaches its threshold
* **medium** - at least one class reaches its threshold
* **high** - at least one class reaches t + 0.25 x (1 - t)

| class | threshold | precision | recall | f1 | pr_auc | support |
|---|---|---|---|---|---|---|
| access-control | 0.419 | 0.621 | 0.813 | 0.704 | 0.756 | 1,437 |
| arithmetic | 0.311 | 0.689 | 0.859 | 0.765 | 0.854 | 1,938 |
| bad-randomness | 0.276 | 0.761 | 0.823 | 0.791 | 0.860 | 402 |
| locked-ether | 0.306 | 0.575 | 0.859 | 0.689 | 0.827 | 553 |
| other | 0.092 | 0.658 | 0.937 | 0.773 | 0.915 | 3,916 |
| reentrancy | 0.248 | 0.592 | 0.915 | 0.719 | 0.862 | 2,853 |
| unchecked-calls | 0.107 | 0.785 | 0.957 | 0.863 | 0.962 | 4,417 |

(per-class precision/recall/F1/PR-AUC above are on the held-out **test** split)

## Evaluation (held-out test split, cluster-aware)
| model | macro_pr_auc | macro_pr_auc_ci | macro_f1 | macro_recall | micro_f1 |
|---|---|---|---|---|---|
| Majority / prior baseline | 0.227 | [0.193, 0.261] | 0.346 | 1.000 | 0.370 |
| Logistic regression | 0.837 | [0.785, 0.866] | 0.737 | 0.905 | 0.757 |
| Decision tree | 0.584 | [0.482, 0.647] | 0.509 | 0.838 | 0.568 |
| Random forest | 0.820 | [0.747, 0.856] | 0.680 | 0.870 | 0.720 |
| K-nearest neighbours | 0.697 | [0.605, 0.746] | 0.587 | 0.788 | 0.644 |
| MLP neural network | 0.862 | [0.811, 0.889] | 0.758 | 0.880 | 0.776 |

## Independent check: SmartBugs Curated (recall, 95 % bootstrap interval)
42 of the 143 SmartBugs contracts also occur, exactly or nearly, in the training pool (found after the freeze).
Pooled recall over the confirmed categories: **52.9%
on the 51 contracts with no overlap** (the independent number),
62.6% on all of them.

| smartbugs_category | mapped_class | status | contracts | recall | ci_low | ci_high | contracts_no_overlap | recall_no_overlap |
|---|---|---|---|---|---|---|---|---|
| unchecked_low_level_calls | unchecked-calls | confirmed | 52 | 0.481 | 0.346 | 0.615 | 32 | 0.406 |
| reentrancy | reentrancy | confirmed | 31 | 0.871 | 0.742 | 0.968 | 13 | 0.769 |
| bad_randomness | bad-randomness | confirmed | 8 | 0.625 | 0.372 | 0.875 | 6.000 | 0.667 |
| access_control | access-control | partial | 18 | 0.611 | 0.389 | 0.833 | 18 | 0.611 |
| arithmetic | - | not evaluated | 15 | - | - | - | - | - |
| denial_of_service | - | not evaluated | 6 | - | - | - | - | - |
| time_manipulation | - | not evaluated | 5 | - | - | - | - | - |
| front_running | - | not evaluated | 4 | - | - | - | - | - |
| other | - | not evaluated | 3 | - | - | - | - | - |
| short_addresses | - | not evaluated | 1 | - | - | - | - | - |

## Limitations
* Learns to imitate Slither, including Slither's false positives and blind spots.
* Cannot flag weakness types without a training label (integer overflow, DoS, front running, timestamp tricks).
* Lower recall on SmartBugs (short, mostly Solidity 0.4) and on very small test contracts than on the training
  distribution (E3, E5); the length effect is correlational (long contracts carry more, easier labels).
* The n-gram ablation (E4) suggests 1-2-grams would generalise better to SmartBugs than the frozen 1-3-grams.
* A low risk level does not mean the contract is safe; false negatives are the main risk.
* Line highlights explain the *model*, not the vulnerability; they can point at correlated code.
