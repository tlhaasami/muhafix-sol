"""
Smoke tests for the model families on a tiny synthetic multi-label problem.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from src.evaluate import pr_auc
from src.models import ForestModel, KNNModel, LogRegModel, MLPModel, PriorModel, TreeModel, top_k_cosine


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n, d = 400, 30
    X = rng.random((n, d)).astype(np.float32)
    Y = np.column_stack([(X[:, 0] > 0.6), (X[:, 1] + X[:, 2] > 1.2), (X[:, 3] > 0.85)]).astype(int)
    return sp.csr_matrix(X), Y


@pytest.mark.parametrize("model", [LogRegModel(C=4.0, n_jobs=1), TreeModel(max_depth=6),
                                   ForestModel(n_estimators=30, n_jobs=1)])
def test_models_learn_signal(data, model):
    X, Y = data
    P = model.fit(X[:300], Y[:300]).predict_proba(X[300:])
    assert P.shape == (100, 3) and np.all((P >= 0) & (P <= 1))
    assert np.nanmean(pr_auc(Y[300:], P)) > 0.6


def test_prior_baseline_is_constant(data):
    X, Y = data
    P = PriorModel().fit(X, Y).predict_proba(X[:5])
    assert np.allclose(P, Y.mean(0))


def test_knn_and_top_k():
    D = np.eye(4, dtype=np.float32)
    idx, sim = top_k_cosine(np.array([[1, 0.1, 0, 0]], np.float32) / 1.005, D, 2)
    assert idx[0, 0] == 0 and sim[0, 0] > sim[0, 1]
    Y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    knn = KNNModel(k=1).fit(D, Y)
    assert np.allclose(knn.predict_proba(D), Y)


def test_mlp_trains_and_numpy_forward_matches(data):
    X, Y = data
    m = MLPModel(hidden=(16, 8), max_epochs=15, patience=15, batch_size=32, lr=1e-2, verbose=False, seed=1)
    m.fit(X[:300], Y[:300], X[300:], Y[300:])
    assert len(m.history.train_loss) == 15 and m.history.train_loss[-1] < m.history.train_loss[0]
    P = m.predict_proba(X[300:])
    assert np.nanmean(pr_auc(Y[300:], P)) > 0.6
    P16 = m.compact().predict_proba(X[300:])
    assert np.allclose(P, P16, atol=2e-2)
