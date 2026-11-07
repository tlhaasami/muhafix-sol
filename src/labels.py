"""
Muhafiz.sol - label mapping and target-class selection (blueprint P1.2 / P1.6).

We start from the raw Slither detector names stored with every contract and map
them to label families with the dataset's own ``label_mappings.json`` (the file
the Hugging Face loading script uses), with one documented correction:

* ``unchecked-send`` -> ``unchecked-calls`` (upstream: ``locked-ether``).
  Slither's unchecked-send detector reports an ignored ``send()`` return value,
  which is SWC-104 / DASP "Unchecked Low Level Calls", not locked ether.

Contracts whose Slither run did not succeed are *excluded*: the upstream
script would label them "safe" because their result list is empty.

Upstream id semantics (resolves the "label 4" question, see docs/label_decision.md):
  big / small configs: names are sorted families; id 4 == "safe"
  all config         : 38 detector names + "safe" at id 38; id 4 == "divide-before-multiply"
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

# Class-label name lists exactly as defined in slither-audited-smart-contracts.py
UPSTREAM_LABELS: Dict[str, List[str]] = {
    "all": [
        "uninitialized-state", "constant-function-asm", "locked-ether",
        "incorrect-shift", "divide-before-multiply", "unused-return",
        "write-after-write", "reentrancy-no-eth", "unchecked-lowlevel",
        "incorrect-equality", "weak-prng", "arbitrary-send",
        "uninitialized-local", "reentrancy-eth", "shadowing-abstract",
        "controlled-delegatecall", "unchecked-transfer", "erc20-interface",
        "controlled-array-length", "tautology", "shadowing-state",
        "tx-origin", "unprotected-upgrade", "suicidal",
        "boolean-cst", "unchecked-send", "msg-value-loop",
        "erc721-interface", "constant-function-state", "delegatecall-loop",
        "mapping-deletion", "reused-constructor", "uninitialized-storage",
        "public-mappings-nested", "array-by-reference", "backdoor",
        "rtlo", "name-reused", "safe"],
    "big": ["access-control", "arithmetic", "other", "reentrancy", "safe", "unchecked-calls"],
    "small": ["access-control", "arithmetic", "other", "reentrancy", "safe", "unchecked-calls",
              "locked-ether", "bad-randomness", "double-spending"],
}

OVERRIDES = {"unchecked-send": "unchecked-calls"}

# SmartBugs category -> (our class, mapping status). Only "confirmed" pairs are
# headline results; "partial" pairs are reported separately with a caveat.
SMARTBUGS_MAPPING = {
    "reentrancy": ("reentrancy", "confirmed",
                   "Slither reentrancy-eth / reentrancy-no-eth detect the same pattern."),
    "unchecked_low_level_calls": ("unchecked-calls", "confirmed",
                                  "unchecked-lowlevel and unchecked-send detect ignored call/send results."),
    "bad_randomness": ("bad-randomness", "confirmed",
                       "weak-prng flags randomness derived from block variables."),
    "access_control": ("access-control", "partial",
                       "suicidal / tx-origin / arbitrary-send cover only part of SmartBugs access control "
                       "(e.g. wrong constructor names are not detected by any training label)."),
    "arithmetic": (None, "not evaluated",
                   "Training label 'arithmetic' is divide-before-multiply only; SmartBugs means integer overflow."),
    "denial_of_service": (None, "not evaluated", "No Slither label in the dataset detects DoS."),
    "front_running": (None, "not evaluated", "No Slither label in the dataset detects front running."),
    "time_manipulation": (None, "not evaluated",
                          "Slither's 'timestamp' detector is not among the 38 dataset labels."),
    "short_addresses": (None, "not evaluated", "No corresponding label."),
    "other": (None, "not evaluated", "Heterogeneous category."),
}


def family_map(upstream_mapping: Dict[str, str]) -> Dict[str, str]:
    """Detector -> family, with our documented override applied."""
    out = dict(upstream_mapping)
    out.update(OVERRIDES)
    return out


def families_for(detectors: Iterable[str], mapping: Dict[str, str]) -> List[str]:
    """Label families for one contract ('ignore' dropped; empty -> ['safe'])."""
    fams = {mapping.get(d, "unknown") for d in detectors} - {"ignore", "safe"}
    return sorted(fams) if fams else ["safe"]


def upstream_ids(detectors: Sequence[str], config: str, upstream_mapping: Dict[str, str]) -> List[int]:
    """Re-create the integer ids the upstream loader would emit (for the label-4 audit)."""
    names = UPSTREAM_LABELS[config]
    if not detectors:
        return [names.index("safe")]
    if config == "all":
        return sorted({names.index(d) for d in detectors})
    fams = sorted({upstream_mapping[d] for d in detectors} - {"ignore"})
    if not fams:
        fams = ["safe"]
    return sorted({names.index(f) for f in fams if f in names})


def select_classes(family_lists: Sequence[Sequence[str]], min_prevalence: float,
                   min_train_positives: int, train_fraction: float) -> Dict[str, object]:
    """Apply the rare-class rule on the de-duplicated pool.

    A family is kept when its prevalence >= ``min_prevalence`` and its expected
    number of training positives (prevalence * N * train_fraction) is at least
    ``min_train_positives``. Rare families are merged into 'other' (if 'other'
    itself survives) or excluded.
    """
    n = len(family_lists)
    counts: Dict[str, int] = {}
    for fams in family_lists:
        for f in fams:
            counts[f] = counts.get(f, 0) + 1
    stats = {f: {"positives": c, "prevalence": c / n} for f, c in sorted(counts.items())}
    candidates = [f for f in counts if f not in ("safe", "unknown")]
    kept = [f for f in candidates
            if counts[f] / n >= min_prevalence and counts[f] * train_fraction >= min_train_positives]
    rare = sorted(set(candidates) - set(kept))
    merged_into_other = rare if "other" in kept else []
    excluded = [] if "other" in kept else rare
    return {"classes": sorted(kept), "merged_into_other": merged_into_other,
            "excluded": excluded, "family_stats": stats, "n": n}


def encode(family_lists: Sequence[Sequence[str]], classes: Sequence[str],
           merged_into_other: Sequence[str]) -> np.ndarray:
    """Multi-hot matrix (n, C). Merged rare families switch on 'other'."""
    idx = {c: i for i, c in enumerate(classes)}
    merged = set(merged_into_other)
    y = np.zeros((len(family_lists), len(classes)), dtype=np.int8)
    for r, fams in enumerate(family_lists):
        for f in fams:
            if f in idx:
                y[r, idx[f]] = 1
            elif f in merged and "other" in idx:
                y[r, idx["other"]] = 1
    return y
