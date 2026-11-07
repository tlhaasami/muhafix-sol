"""
Unit and Acceptance Tests for Deduplication and Group-Aware Splits
Covers Work Package P1.5 and Acceptance Tests A2 and A4.
Owner: Talha Sami (24I-3118)
"""

import pytest
from src.splits import cluster_contracts, create_group_aware_splits, compute_sha256


@pytest.fixture
def sample_contracts():
    # 6 mock contracts:
    # c0 and c1 are exact duplicates
    # c2 is a near clone of c0 (just variable rename)
    # c3, c4, c5 are distinct contracts
    return [
        {
            "id": "c0",
            "source_code": "contract Token { uint256 public supply; function get() public view returns(uint256) { return supply; } }"
        },
        {
            "id": "c1",
            "source_code": "contract Token { uint256 public supply; function get() public view returns(uint256) { return supply; } }"
        },
        {
            "id": "c2",
            "source_code": "contract MyToken { uint256 public supply; function get() public view returns(uint256) { return supply; } }"
        },
        {
            "id": "c3",
            "source_code": "contract ReentrancyBug { mapping(address => uint) b; function withdraw() public { msg.sender.call.value(b[msg.sender])(''); b[msg.sender]=0; } }"
        },
        {
            "id": "c4",
            "source_code": "contract AccessControl { address owner; function destroy() public { require(tx.origin == owner); selfdestruct(payable(owner)); } }"
        },
        {
            "id": "c5",
            "source_code": "contract SafeMath { function add(uint a, uint b) internal pure returns (uint) { return a + b; } }"
        }
    ]


def test_exact_and_near_clustering(sample_contracts):
    clustered, stats = cluster_contracts(sample_contracts, jaccard_threshold=0.75)
    assert stats["total_contracts"] == 6
    assert stats["exact_duplicates"] >= 1
    
    # c0 and c1 must have identical group_id
    assert clustered[0]["group_id"] == clustered[1]["group_id"]
    # c0 and c2 (near-duplicates) should share the same group
    assert clustered[0]["group_id"] == clustered[2]["group_id"]
    # c3 should have a distinct group
    assert clustered[0]["group_id"] != clustered[3]["group_id"]


def test_a2_no_group_leakage(sample_contracts):
    """
    Acceptance Test A2: No group ID appears in more than one split.
    """
    clustered, _ = cluster_contracts(sample_contracts, jaccard_threshold=0.75)
    train, val, test, _ = create_group_aware_splits(clustered, seed=42, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

    train_groups = {c["group_id"] for c in train}
    val_groups = {c["group_id"] for c in val}
    test_groups = {c["group_id"] for c in test}

    assert train_groups.isdisjoint(val_groups), "Group leakage between train and val!"
    assert train_groups.isdisjoint(test_groups), "Group leakage between train and test!"
    assert val_groups.isdisjoint(test_groups), "Group leakage between val and test!"


def test_a4_determinism(sample_contracts):
    """
    Acceptance Test A4: Running the split twice with the same seed produces identical split hashes.
    """
    clustered1, _ = cluster_contracts([dict(c) for c in sample_contracts], jaccard_threshold=0.75)
    train1, val1, test1, rand1 = create_group_aware_splits(clustered1, seed=42)

    clustered2, _ = cluster_contracts([dict(c) for c in sample_contracts], jaccard_threshold=0.75)
    train2, val2, test2, rand2 = create_group_aware_splits(clustered2, seed=42)

    hash_train1 = compute_sha256("".join(c["id"] for c in train1))
    hash_train2 = compute_sha256("".join(c["id"] for c in train2))

    assert hash_train1 == hash_train2, "Splits were not deterministic!"
