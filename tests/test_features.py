"""
Tests for src/features.py (blueprint P1.7).
"""

import numpy as np

from src.features import FeaturePipeline, NgramAnalyzer

TRAIN = [
    "function withdraw ( ) public { msg.sender . call . value ( x ) ( ) ; }",
    "function deposit ( ) public payable { balances [ msg.sender ] += msg.value ; }",
    "function kill ( ) public { selfdestruct ( owner ) ; }",
    "require ( tx.origin == owner ) ; owner . transfer ( this . balance ) ;",
] * 5
UNSEEN = "unseen_identifier_xyz unseen_identifier_xyz"


def _fit(**kw):
    kw = {"ngram_range": (1, 3), "max_features": 200, "min_df": 1, "max_df": 1.0, "n_jobs": 1, **kw}
    return FeaturePipeline(**kw).fit(TRAIN)


def test_analyzer_ngrams_and_ids():
    a = NgramAnalyzer((1, 2))
    assert a("a b c") == ["a", "b", "c", "a b", "b c"]
    assert NgramAnalyzer((1, 1), ids=True)("balance x . call msg.sender") == \
        ["balance", "<ID>", ".", "call", "msg.sender"]


def test_vocabulary_comes_from_train_only():
    pipe = _fit()
    assert "unseen_identifier_xyz" not in pipe.vocabulary
    assert pipe.transform([UNSEEN]).nnz == 0
    assert len(pipe.vocabulary) <= 200 and "msg.sender . call" in pipe.vocabulary


def test_transform_is_deterministic_and_l2_normalised():
    pipe = _fit()
    a = pipe.transform([TRAIN[0]]).toarray()
    b = pipe.transform([TRAIN[0]]).toarray()
    assert np.array_equal(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5


def test_max_df_removes_ubiquitous_tokens():
    pipe = _fit(max_df=0.9)
    assert "(" not in pipe.vocabulary  # appears in every training document


def test_parallel_matches_serial():
    serial = _fit(n_jobs=1).transform(TRAIN * 30).toarray()
    parallel = _fit(n_jobs=2).transform(TRAIN * 30).toarray()
    assert np.allclose(serial, parallel)


def test_save_load_roundtrip(tmp_path):
    pipe = _fit(svd_components=3)
    p = tmp_path / "vec.joblib"
    pipe.save(p)
    again = FeaturePipeline.load(p)
    X1, X2 = pipe.transform(TRAIN[:3]), again.transform(TRAIN[:3])
    assert np.allclose(X1.toarray(), X2.toarray())
    assert again.transform_svd(X2).shape == (3, 3)
    assert pipe.benchmark_single(TRAIN[0]) < 1000
