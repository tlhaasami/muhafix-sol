"""
Unit tests for src/splits.py (blueprint P1.5; acceptance tests A2 and A4 on toy
data - the same assertions run on the real data in tests/test_data_pack.py).
"""

import numpy as np
import pytest

from src.preprocess import prepare_tokens
from src.splits import (
    UnionFind,
    cluster,
    dedup_tokens,
    exact_hash,
    group_stratified_split,
    leakage_rate,
    minhash_signatures,
    random_split,
    shingles,
)

BASE = """
pragma solidity ^0.4.24;
contract Bank {
    mapping(address => uint) public balances;
    function deposit() public payable { balances[msg.sender] += msg.value; }
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount);
        msg.sender.call.value(amount)();
        balances[msg.sender] -= amount;
    }
    function total() public view returns (uint) { return address(this).balance; }
}
"""


def _toks(src):
    return dedup_tokens(prepare_tokens(src))


@pytest.fixture
def corpus():
    renamed = BASE.replace("Bank", "Vault").replace("balances", "credit").replace("amount", "wad")
    near = BASE.replace("function total() public view returns (uint) { return address(this).balance; }",
                        "function total() public view returns (uint) { return 42; }")
    other1 = "contract Token { uint public supply = 100; function burn(uint x) public { supply -= x; } }"
    other2 = "contract Lottery { function play() public payable { if (block.timestamp % 2 == 0) msg.sender.transfer(1); } }"
    return [BASE, BASE, renamed, near, other1, other2]


def test_union_find_components():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 4)
    uf.union(1, 4)
    lab = uf.labels()
    assert lab[0] == lab[1] == lab[3] == lab[4] and lab[2] != lab[0]


def test_shingles_short_and_long():
    assert shingles([]) == []
    assert shingles(["a", "b"]) == [b"a b"]
    assert len(shingles(list("abcdefg"), k=5)) == 3


def test_renamed_copy_is_exact_duplicate(corpus):
    assert exact_hash(_toks(corpus[0])) == exact_hash(_toks(corpus[1])) == exact_hash(_toks(corpus[2]))
    assert exact_hash(_toks(corpus[0])) != exact_hash(_toks(corpus[3]))


def test_cluster_groups_exact_and_near_duplicates(corpus):
    toks = [_toks(c) for c in corpus]
    groups, stats = cluster([exact_hash(t) for t in toks], minhash_signatures(toks), threshold=0.8)
    assert groups[0] == groups[1] == groups[2] == groups[3]       # clones together
    assert len({groups[0], groups[4], groups[5]}) == 3            # distinct contracts apart
    assert stats["exact_duplicates"] == 2 and stats["largest_group"] == 4


def test_split_has_no_group_leakage_and_is_deterministic():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), rng.integers(1, 6, 400))
    labels = (rng.random((len(groups), 4)) < [0.3, 0.1, 0.05, 0.5]).astype(int)
    s1 = group_stratified_split(groups, labels, seed=42)
    s2 = group_stratified_split(groups, labels, seed=42)
    assert np.array_equal(s1, s2)                                  # A4 determinism
    sets = [set(groups[s1 == k]) for k in range(3)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])  # A2
    frac = np.bincount(s1, minlength=3) / len(s1)
    assert np.allclose(frac, [0.70, 0.15, 0.15], atol=0.03)
    for k in range(3):                                             # A5 coverage
        assert (labels[s1 == k].sum(0) > 0).all()


def test_giant_group_goes_to_train():
    groups = np.concatenate([np.zeros(300, int), np.arange(1, 701)])
    labels = np.ones((1000, 1), int)
    split = group_stratified_split(groups, labels, seed=1)
    assert (split[:300] == 0).all()


def test_groups_above_cap_are_forced_into_train():
    groups = np.concatenate([np.zeros(50, int), np.ones(50, int), np.arange(2, 902)])
    labels = np.ones((1000, 1), int)
    split = group_stratified_split(groups, labels, seed=3, max_eval_group_frac=0.04)
    assert (split[:100] == 0).all()                 # both 50-member groups exceed 4 % of 1,000
    assert set(np.unique(split[100:])) == {0, 1, 2}


def test_random_split_sizes_and_leakage_measure():
    split = random_split(100, (70, 15, 15), seed=3)
    assert np.bincount(split).tolist() == [70, 15, 15]
    groups = np.repeat(np.arange(50), 2)
    assert leakage_rate(groups, split) > 0.3          # random split leaks clones
    clean = group_stratified_split(groups, np.ones((100, 1), int), seed=3)
    assert leakage_rate(groups, clean) == 0.0
