"""
Muhafiz.sol - de-duplication and group-aware splits (blueprint P1.5).

1. Exact duplicates: SHA-256 of the normalised token string (identifiers
   replaced by <ID>, so renamed copies count as exact duplicates). One
   representative per hash is kept; the group size is remembered.
2. Near duplicates: MinHash signatures of token 5-gram shingles (datasketch),
   locality-sensitive hashing to find candidate pairs, the MinHash estimate of
   Jaccard >= 0.8 to confirm a pair, and connected components (union-find) as
   clone groups.
3. Group-aware split: whole groups go to train / validation / test (70/15/15)
   with a greedy multi-label stratification so every class has positives in
   every split.
4. Random split of identical sizes over the same contracts, for experiment E2.

The functions are pure (lists/arrays in, arrays out) so they can be unit-tested
on toy data; ``build_data`` wires them to the real dataset.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.preprocess import normalise_identifiers

SPLITS = ("train", "val", "test")
MINHASH_SCHEME = "affine32"  # datasketch >= 2.0 requires the scheme to be explicit


# --------------------------------------------------------------------------- #
# Union-find
# --------------------------------------------------------------------------- #
class UnionFind:
    """Array-based union-find with path halving and union by size."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Hashing and MinHash
# --------------------------------------------------------------------------- #
def dedup_tokens(tokens: Sequence[str]) -> List[str]:
    """Token view used for clone detection: identifiers anonymised."""
    return normalise_identifiers(list(tokens))


def exact_hash(tokens: Sequence[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def shingles(tokens: Sequence[str], k: int = 5) -> List[bytes]:
    """Set of token k-gram shingles, encoded as bytes for hashing."""
    if not tokens:
        return []
    if len(tokens) < k:
        return [" ".join(tokens).encode("utf-8")]
    return list({" ".join(tokens[i:i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)})


def minhash_signature(tokens: Sequence[str], k: int = 5, num_perm: int = 128,
                      seed: int = 1) -> np.ndarray:
    """MinHash signature (datasketch) of the token k-gram shingle set."""
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm, seed=seed, scheme=MINHASH_SCHEME)
    sh = shingles(tokens, k)
    if sh:
        m.update_batch(sh)
    return m.hashvalues.astype(np.uint64)


def _signature_job(args: Tuple[List[str], int, int]) -> np.ndarray:
    tokens, k, num_perm = args
    return minhash_signature(tokens, k, num_perm)


def minhash_signatures(token_lists: Iterable[Sequence[str]], k: int = 5, num_perm: int = 128,
                       n_jobs: int = 1) -> np.ndarray:
    """Signatures for many documents, optionally in parallel processes."""
    jobs = [(list(t), k, num_perm) for t in token_lists]
    if n_jobs == 1 or len(jobs) < 200:
        sigs = [_signature_job(j) for j in jobs]
    else:
        from joblib import Parallel, delayed

        sigs = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_signature_job)(j) for j in jobs)
    return np.vstack(sigs) if sigs else np.zeros((0, num_perm), dtype=np.uint64)


def near_duplicate_pairs(signatures: np.ndarray, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
    """Candidate pairs from MinHash LSH, confirmed by the signature Jaccard estimate."""
    from datasketch import LeanMinHash, MinHashLSH

    n, num_perm = signatures.shape
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    lean = []
    for i in range(n):
        lm = LeanMinHash(seed=1, hashvalues=signatures[i], scheme=MINHASH_SCHEME)
        lean.append(lm)
        lsh.insert(i, lm)
    pairs = []
    for i in range(n):
        for j in lsh.query(lean[i]):
            if j > i:
                est = float(np.mean(signatures[i] == signatures[j]))
                if est >= threshold:
                    pairs.append((i, int(j), est))
    return pairs


def cluster(exact_hashes: Sequence[str], signatures: np.ndarray, threshold: float = 0.8
            ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Group documents: exact-hash groups first, then near-duplicate components.

    ``signatures`` must be aligned with ``exact_hashes`` (one row per document).
    Returns a dense group id per document and summary statistics.
    """
    n = len(exact_hashes)
    uf = UnionFind(n)
    first: Dict[str, int] = {}
    for i, h in enumerate(exact_hashes):
        if h in first:
            uf.union(first[h], i)
        else:
            first[h] = i
    reps = np.array(sorted(first.values()), dtype=np.int64)
    pairs = near_duplicate_pairs(signatures[reps], threshold) if len(reps) > 1 else []
    for a, b, _ in pairs:
        uf.union(int(reps[a]), int(reps[b]))
    roots = uf.labels()
    _, group = np.unique(roots, return_inverse=True)
    sizes = np.bincount(group)
    stats = {
        "documents": n,
        "unique_exact": int(len(reps)),
        "exact_duplicates": int(n - len(reps)),
        "exact_duplicate_pct": 100.0 * (n - len(reps)) / max(n, 1),
        "near_duplicate_pairs": len(pairs),
        "groups": int(len(sizes)),
        "singleton_groups": int((sizes == 1).sum()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "docs_in_multi_member_groups_pct": 100.0 * sizes[sizes > 1].sum() / max(n, 1),
    }
    return group.astype(np.int64), stats


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def group_stratified_split(groups: np.ndarray, labels: np.ndarray,
                           ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                           seed: int = 42, max_eval_group_frac: float = 0.0) -> np.ndarray:
    """Assign whole groups to train/val/test with greedy multi-label stratification.

    Groups larger than ``max_eval_group_frac * n`` (giant template families) are
    always put in train: otherwise a single family of ~1,000 near-identical
    contracts can make up 10 % of the validation or test split, so thresholds get
    tuned to - and scores measured on - one template instead of on new contracts.

    The remaining groups are visited from the largest (then rarest-label) to the
    smallest; each goes to the split with the largest remaining *deficit*,
    measured on the group's labels (weighted by label rarity) and on the document
    count. This is the iterative-stratification idea of Sechidis et al. (2011)
    applied to groups. Returns split indices (0 = train, 1 = val, 2 = test).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.float64)
    n, c = labels.shape
    n_groups = int(groups.max()) + 1 if n else 0
    g_size = np.bincount(groups, minlength=n_groups).astype(np.float64)
    g_lab = np.zeros((n_groups, c))
    np.add.at(g_lab, groups, labels)

    r = np.asarray(ratios, dtype=np.float64) / np.sum(ratios)
    want_docs = r * n
    want_lab = np.outer(r, labels.sum(0))                      # (3, C)
    rarity = 1.0 / np.maximum(labels.sum(0), 1.0)              # rare labels weigh more
    have_docs = np.zeros(3)
    have_lab = np.zeros((3, c))

    order = rng.permutation(n_groups)                          # random tie-break
    rare_score = (g_lab > 0) @ rarity
    order = order[np.lexsort((-rare_score[order], -g_size[order]))]

    assign = np.zeros(n_groups, dtype=np.int64)
    if max_eval_group_frac > 0:
        giant = g_size > max_eval_group_frac * n
        have_docs[0] += g_size[giant].sum()
        have_lab[0] += g_lab[giant].sum(0)
        order = order[~giant[order]]
    for g in order:
        lab_def = ((want_lab - have_lab) / np.maximum(want_lab, 1.0)) @ (g_lab[g] * rarity)
        doc_def = (want_docs - have_docs) / np.maximum(want_docs, 1.0)
        # a split may not exceed its document target unless it is train
        too_big = have_docs + g_size[g] > want_docs * 1.02
        too_big[0] = False
        score = lab_def + doc_def - 1e6 * too_big
        s = int(np.argmax(score))
        assign[g] = s
        have_docs[s] += g_size[g]
        have_lab[s] += g_lab[g]
    return assign[groups]


def random_split(n: int, sizes: Tuple[int, int, int], seed: int = 42) -> np.ndarray:
    """Random split with exactly the given split sizes (experiment E2)."""
    if sum(sizes) != n:
        raise ValueError("sizes must add up to n")
    perm = np.random.default_rng(seed).permutation(n)
    out = np.empty(n, dtype=np.int64)
    out[perm[:sizes[0]]] = 0
    out[perm[sizes[0]:sizes[0] + sizes[1]]] = 1
    out[perm[sizes[0] + sizes[1]:]] = 2
    return out


def leakage_rate(groups: np.ndarray, split: np.ndarray) -> float:
    """Share of non-train documents whose clone group also appears in train."""
    train_groups = set(groups[split == 0].tolist())
    other = split != 0
    if not other.any():
        return 0.0
    return float(np.mean([g in train_groups for g in groups[other]]))
