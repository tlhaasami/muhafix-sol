# Label Mapping and Classification Decision Document
**Muhafiz.sol — Smart Contract Vulnerability Detector**  
**Author:** Talha Sami (24I-3118), Phase 1 Owner  
**Date:** 27 October 2026  
**Status:** Completed & Validated against Upstream Source  

---

## 1. Executive Summary & Purpose
This document formally records the investigation and final resolution of:
1. The exact semantic meaning of **Label 4** in the Slither Audited Smart Contracts dataset.
2. The meaning of label IDs above 5 and up to 38 across different dataset configurations (`big-multilabel`, `small-multilabel`, `all-multilabel`).
3. The agreed mapping rule from raw Slither detectors and SmartBugs categories to our target multi-label classification classes.

---

## 2. Upstream Dataset Ground Truth

Inspection of `slither-audited-smart-contracts.py` and `data/label_mappings.json` directly from the Hugging Face repository (`mwritescode/slither-audited-smart-contracts`) reveals that label IDs correspond to three distinct configuration schema definitions:

```python
_LABELS = {
    'big': [
        'access-control',      # ID 0
        'arithmetic',          # ID 1
        'other',               # ID 2
        'reentrancy',          # ID 3
        'safe',                # ID 4
        'unchecked-calls'      # ID 5
    ],
    'small': [
        'access-control',      # ID 0
        'arithmetic',          # ID 1
        'other',               # ID 2
        'reentrancy',          # ID 3
        'safe',                # ID 4
        'unchecked-calls',     # ID 5
        'locked-ether',        # ID 6
        'bad-randomness',      # ID 7
        'double-spending'      # ID 8
    ],
    'all': [
        # 38 individual Slither detector names + 'safe' at ID 38
    ]
}
```

---

## 3. Resolution of Key Architectural Questions

### A. What does Label 4 mean?
* **Finding:** In both `big` and `small` configurations, **Label 4 corresponds exactly to `'safe'`**.
* **Semantics:** A contract tagged with Label 4 has passed Slither static analysis with **zero detected vulnerabilities**.
* **Implication for Multi-Label Modeling:** In a multi-label vulnerability classifier, a vulnerability class represents the presence of a flaw. If a contract is labelled with any vulnerability class (0, 1, 2, 3, 5, 6, 7, 8), it should NOT be flagged as safe.
* **Decision:**
  * During multi-label target encoding, we do **not** treat `safe` as a positive vulnerability class.
  * A contract is determined to be **safe** when all predicted vulnerability class probabilities fall below their tuned operational thresholds ($\hat{y}_c = 0$ for all classes $c$).
  * This prevents contradictory outputs (e.g., predicting `reentrancy = 1` and `safe = 1` simultaneously).

### B. What do IDs above 5 (or above 9 in uncollapsed configurations) represent?
* In `small-multilabel`, IDs 6, 7, and 8 represent specific high-impact vulnerability categories:
  * ID 6: `locked-ether` (contracts holding ether that cannot be withdrawn)
  * ID 7: `bad-randomness` (reliance on predictable block variables like `block.timestamp` / `block.number`)
  * ID 8: `double-spending` (`msg.value` in loop or repeated delegatecalls)
* In `all-multilabel`, IDs 0 to 37 represent raw fine-grained Slither detector names (e.g., `suicidal`, `arbitrary-send`, `unchecked-lowlevel`, `shadowing-state`), while ID 38 represents `safe`.

---

## 4. Final Target Class Set Selection

Applying the blueprint rule ($\ge 1\%$ prevalence and sufficient positive instances in training), our unified target classes for Muhafiz.sol are:

| Class Name | Target ID | Mapped Slither Detectors | SmartBugs Equivalent Category |
| :--- | :---: | :--- | :--- |
| **access-control** | 0 | `suicidal`, `arbitrary-send`, `controlled-delegatecall`, `unprotected-upgrade`, `tx-origin` | `access_control` |
| **arithmetic** | 1 | `divide-before-multiply` (and integer under/overflow) | `arithmetic` |
| **reentrancy** | 2 | `reentrancy-eth`, `reentrancy-no-eth` | `reentrancy` |
| **unchecked-calls**| 3 | `unused-return`, `unchecked-lowlevel`, `unchecked-transfer` | `unchecked_low_level_calls` |
| **locked-ether** | 4 | `locked-ether`, `unchecked-send` | `denial_of_service` / `other` |
| **bad-randomness** | 5 | `weak-prng` | `bad_randomness` |
| **other / unknown**| 6 | Miscellaneous low-frequency warnings | Unmapped categories |

Any detector categorized under `"ignore"` in `label_mappings.json` (such as cosmetic shadowing or standard interface warnings) is filtered out and excluded from vulnerability scoring.

---

## 5. Teammate Sign-off
* **Phase 1 Owner (Data):** Talha Sami (24I-3118) — *Proposed & Verified*
* **Phase 2 Owner (Model):** Sohaib (24I-3108) — *Reviewed for Experiment E3*
* **Phase 3 Owner (App/UI):** Shaiman (24I-3074) — *Reviewed for Explanation & Triage UI*
