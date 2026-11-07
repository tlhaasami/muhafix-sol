# Label decision (Gate G1 record)

**Project:** Muhafiz.sol · **Phase 1 owner:** Talha Sami (24I-3118) · **Decision date:** 8 November 2026
**Evidence:** `reports/tables/label4_audit.csv`, `reports/tables/family_distribution.csv`,
`reports/tables/detector_frequency.csv`, `notebooks/01_data_audit.ipynb`, `models/label_map.json`

## 1. What we looked at

The Hugging Face loading script (`slither-audited-smart-contracts.py`) reads nine raw parquet shards that hold,
for every contract, the full Slither JSON output. It then converts the detector names into integer ids in one
of three ways (configurations `all`, `big`, `small`). Instead of trusting the pre-collapsed ids we re-derived
every label from the raw detector names (`src/labels.py`).

## 2. What does label 4 mean? (blueprint risk R1)

| Configuration | id 4 is | `safe` is | id 4 together with other ids? | highest id |
|---|---|---|---|---|
| `big` / `small` | `safe` | id 4 | never | 5 (big), 8 (small) |
| `all` | `divide-before-multiply` | id 38 | yes | 38 |

The loader uses a different name list per configuration. In `big`/`small`, id 4 is `safe` and only appears
alone. In `all` (the configuration shown by default in the dataset viewer), id 4 is the
`divide-before-multiply` detector and `safe` is id 38. So the viewer rows where 4 appears with other ids and
ids reach 38 come from the `all` configuration. This **resolves the contradiction noted in the blueprint**:
both statements were true, but about different configurations. The exact shares are in
`reports/tables/label4_audit.csv`.

**Decision:** `safe` is not a model class. A contract counts as "no flagged weakness" when every class is
below its threshold, so the model can never say "reentrancy" and "safe" at the same time.

## 3. Failed Slither runs

The loader labels a contract `safe` whenever Slither's result list is empty. That would also cover runs where
Slither crashed. We checked the `success` flag of all 120,608 contracts: **0 runs failed**. The dataset
author evidently kept only successful runs. The filter is still in the pipeline in case the source changes.

## 4. Which configuration?

The `big` and `small` split files are disjoint (106,474 + 14,134 contracts). Our audit shows that **every**
contract with a `locked-ether`, `bad-randomness` or `double-spending` finding is in `small`. `big` holds none.
Training on `big-multilabel` alone (the blueprint's first suggestion) would therefore make the SmartBugs *Bad
Randomness* category impossible to evaluate. **Decision:** use all 120,608 contracts (big ∪ small), map
detectors with the upstream `label_mappings.json`, and build our own cluster-aware split. The upstream splits
are not used for results (blueprint 6.4).

## 5. Detector → family mapping

We use the upstream `label_mappings.json` with **one documented correction**:

| Detector | Upstream family | Ours | Reason |
|---|---|---|---|
| `unchecked-send` | `locked-ether` | `unchecked-calls` | Slither's *unchecked-send* reports an ignored `send()` return value: SWC-104 / DASP "Unchecked Low Level Calls", not locked ether. |

Detectors mapped to `ignore` (shadowing, interface conformance, naming) are dropped. A contract whose only
detectors are ignored counts as having no flagged weakness.

## 6. Target classes (rare-class rule)

Rule (config `labels`): keep a family when it reaches **≥ 1 % prevalence** in the de-duplicated pool **and**
**≥ 300 expected training positives**. Otherwise merge it into `other` (if `other` is kept) or exclude it.

* Kept (7): `access-control`, `arithmetic`, `bad-randomness`, `locked-ether`, `other`, `reentrancy`,
  `unchecked-calls`.
* Merged into `other`: `double-spending` (msg-value-loop, delegatecall-loop), about 0.4 % prevalence.

The exact prevalences after de-duplication are in `models/label_map.json` (`family_stats_dedup_pool`) and
`reports/tables/family_distribution.csv`.

## 7. Mapping to SmartBugs Curated (used by experiment E3)

| SmartBugs category | Our class | Status | Why |
|---|---|---|---|
| Reentrancy | reentrancy | **confirmed** | reentrancy-eth / reentrancy-no-eth detect the same pattern |
| Unchecked Low Level Calls | unchecked-calls | **confirmed** | unchecked-lowlevel and unchecked-send detect ignored call/send results |
| Bad Randomness | bad-randomness | **confirmed** | weak-prng flags randomness derived from block variables |
| Access Control | access-control | partial | suicidal / tx-origin / arbitrary-send cover only part of the category (e.g. wrong constructor names have no training label) |
| Arithmetic | – | not evaluated | the training label `arithmetic` is divide-before-multiply only; SmartBugs means integer overflow |
| Denial of Service, Front Running, Time Manipulation, Short Addresses, Other | – | not evaluated | no corresponding training label (Slither's `timestamp` detector is not among the 38 dataset labels) |

Headline SmartBugs results use **confirmed** pairs only. *Access control* is reported separately with its
caveat. No mapping is forced (blueprint 6.3).

## 8. Plan B (blueprint 6.6)

Not triggered: the labels map cleanly to families, so the multi-label target is kept.

## 9. Sign-off

| Member | Role | Status |
|---|---|---|
| Talha Sami (24I-3118) | Phase 1 owner, proposer | proposed |
| Sohaib (24I-3108) | Phase 2 owner (needs mapping for E3) | **pending review** |
| Shaiman (24I-3074) | Phase 3 owner, Gate G1 reviewer | **pending review** |

*Earlier drafts of this file recorded teammate sign-off before any review took place; that has been
corrected. Sign-off must be added by each member after the G1 walkthrough.*
