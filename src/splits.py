"""
Muhafiz.sol - Deduplication and Group-Aware Splits Module
Phase 1: Work Package P1.5
Owner: Talha Sami (24I-3118)

Implements:
1. Exact deduplication via SHA-256 of normalized tokens.
2. Near-duplicate clustering using MinHash and Locality-Sensitive Hashing (LSH)
   with Union-Find connected components (Jaccard threshold >= 0.80).
3. Group-aware train/val/test splitting (70/15/15) preventing clone leakage.
4. Independent random split generator for Experiment E2 (leakage ablation).
"""

import hashlib
import random
import json
import os
from typing import List, Dict, Set, Tuple, Any
from src.preprocess import clean_source, normalise, tokenize


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_token_shingles(tokens: List[str], shingle_size: int = 5) -> Set[str]:
    """Generate k-gram token shingles."""
    if len(tokens) < shingle_size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate exact Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def cluster_contracts(
    contracts: List[Dict[str, Any]],
    jaccard_threshold: float = 0.80,
    shingle_size: int = 5
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform exact and near-duplicate clustering.
    Returns:
    - contracts with added 'group_id' and 'is_representative'
    - deduplication summary statistics
    """
    total_count = len(contracts)
    if total_count == 0:
        return [], {}

    # Step 1: Preprocess and Exact Deduplication (SHA-256)
    exact_hash_groups: Dict[str, List[int]] = {}
    for idx, c in enumerate(contracts):
        clean = clean_source(c.get("source_code", ""))
        norm = normalise(clean)
        toks = tokenize(norm)
        c["_clean"] = clean
        c["_norm"] = norm
        c["_tokens"] = toks
        c["_shingles"] = get_token_shingles(toks, shingle_size)
        c["_hash"] = compute_sha256(" ".join(toks))

        if c["_hash"] not in exact_hash_groups:
            exact_hash_groups[c["_hash"]] = []
        exact_hash_groups[c["_hash"]].append(idx)

    # Representatives of exact hash groups
    unique_indices = [indices[0] for indices in exact_hash_groups.values()]
    exact_duplicates_count = total_count - len(unique_indices)

    # Step 2: Near-duplicate clustering across representatives
    uf = UnionFind()
    for idx in range(total_count):
        uf.find(idx)

    # First merge exact duplicates into same component
    for indices in exact_hash_groups.values():
        first = indices[0]
        for other in indices[1:]:
            uf.union(first, other)

    # MinHash LSH or pairwise candidate checking for near-duplicates
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=128)
        minhashes = {}
        for idx in unique_indices:
            m = MinHash(num_perm=128)
            for shingle in contracts[idx]["_shingles"]:
                m.update(shingle.encode("utf-8"))
            minhashes[idx] = m
            lsh.insert(f"c_{idx}", m)

        for idx in unique_indices:
            candidates = lsh.query(minhashes[idx])
            for cand in candidates:
                cand_idx = int(cand.replace("c_", ""))
                if cand_idx != idx:
                    # Verify exact Jaccard similarity to prevent LSH false positives
                    sim = jaccard_similarity(contracts[idx]["_shingles"], contracts[cand_idx]["_shingles"])
                    if sim >= jaccard_threshold:
                        uf.union(idx, cand_idx)
    except ImportError:
        # Fallback if datasketch is installing: pairwise Jaccard on representatives
        for i in range(len(unique_indices)):
            idx_i = unique_indices[i]
            s_i = contracts[idx_i]["_shingles"]
            for j in range(i + 1, len(unique_indices)):
                idx_j = unique_indices[j]
                s_j = contracts[idx_j]["_shingles"]
                if jaccard_similarity(s_i, s_j) >= jaccard_threshold:
                    uf.union(idx_i, idx_j)

    # Step 3: Assign unified group_id
    groups: Dict[int, List[int]] = {}
    for idx in range(total_count):
        root = uf.find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(idx)

    largest_group_size = max(len(members) for members in groups.values()) if groups else 0

    group_id_counter = 0
    for root, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        for idx in members:
            contracts[idx]["group_id"] = f"group_{group_id_counter}"
            contracts[idx]["is_representative"] = (idx == members[0])
        group_id_counter += 1

    stats = {
        "total_contracts": total_count,
        "exact_duplicates": exact_duplicates_count,
        "exact_duplicate_pct": (exact_duplicates_count / total_count * 100.0) if total_count else 0.0,
        "total_groups": len(groups),
        "largest_group_size": largest_group_size
    }
    return contracts, stats


def create_group_aware_splits(
    contracts: List[Dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split contracts by group_id so all clones of a contract sit strictly in one split.
    Also produces an un-grouped random split of identical size for Experiment E2.
    """
    rng = random.Random(seed)

    # Group contracts by group_id
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in contracts:
        gid = c["group_id"]
        if gid not in group_map:
            group_map[gid] = []
        group_map[gid].append(c)

    group_keys = sorted(list(group_map.keys()))
    rng.shuffle(group_keys)

    total_contracts = len(contracts)
    target_train = int(total_contracts * train_ratio)
    target_val = int(total_contracts * val_ratio)

    train_set = []
    val_set = []
    test_set = []

    # Distribute entire groups to train, val, test
    train_groups_list = []
    val_groups_list = []
    test_groups_list = []

    train_count = 0
    val_count = 0

    for gid in group_keys:
        members = group_map[gid]
        count = len(members)
        if train_count + count <= target_train or not train_groups_list:
            train_groups_list.append(gid)
            train_count += count
        elif val_count + count <= target_val or not val_groups_list:
            val_groups_list.append(gid)
            val_count += count
        else:
            test_groups_list.append(gid)

    # Ensure all splits have at least one group if there are >= 3 groups
    if not test_groups_list and len(group_keys) >= 3:
        if len(val_groups_list) > 1:
            test_groups_list.append(val_groups_list.pop())
        elif len(train_groups_list) > 1:
            test_groups_list.append(train_groups_list.pop())

    train_set = [c for gid in train_groups_list for c in group_map[gid]]
    val_set = [c for gid in val_groups_list for c in group_map[gid]]
    test_set = [c for gid in test_groups_list for c in group_map[gid]]

    # Build un-grouped random split for Experiment E2
    all_contracts_shuffled = list(contracts)
    rng_random = random.Random(seed)
    rng_random.shuffle(all_contracts_shuffled)
    n_train = len(train_set)
    n_val = len(val_set)

    random_train = all_contracts_shuffled[:n_train]
    random_val = all_contracts_shuffled[n_train:n_train + n_val]
    random_test = all_contracts_shuffled[n_train + n_val:]

    # Assertions for Acceptance Test A2 (No Group Leakage)
    train_groups = {c["group_id"] for c in train_set}
    val_groups = {c["group_id"] for c in val_set}
    test_groups = {c["group_id"] for c in test_set}

    assert train_groups.isdisjoint(val_groups), "Leakage detected between train and val groups!"
    assert train_groups.isdisjoint(test_groups), "Leakage detected between train and test groups!"
    assert val_groups.isdisjoint(test_groups), "Leakage detected between val and test groups!"

    return train_set, val_set, test_set, random_test
