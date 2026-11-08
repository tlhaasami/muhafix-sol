"""
Muhafiz.sol - model families compared in experiment E1 (blueprint 8.2, P2.1-P2.3).

Every model exposes ``fit(X, Y)`` and ``predict_proba(X) -> (n, C)`` so the
harness can treat them identically.

  PriorModel      majority / prior baseline (constant class prevalence)
  LogRegModel     one-vs-rest logistic regression, balanced class weights
  TreeModel       decision tree (multi-output, class-weighted)
  ForestModel     random forest (multi-output, balanced_subsample)
  KNNModel        cosine k-nearest neighbours, similarity-weighted votes
  MLPModel        PyTorch MLP: 2 hidden layers, ReLU, dropout, sigmoid outputs,
                  BCE with per-class positive weights, Adam, early stopping on
                  validation macro PR-AUC. Inference is exported to numpy so
                  the app does not need PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from src.evaluate import pr_auc


def _proba_list_to_matrix(probas: List[np.ndarray]) -> np.ndarray:
    """sklearn multi-output predict_proba returns one (n, 2) array per class."""
    cols = []
    for p in probas:
        cols.append(p[:, 1] if p.shape[1] == 2 else np.zeros(p.shape[0]))
    return np.column_stack(cols)


# --------------------------------------------------------------------------- #
class PriorModel:
    name = "prior"

    def fit(self, X, Y):
        self.prior_ = np.asarray(Y).mean(0)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile(self.prior_, (n, 1))


class LogRegModel:
    name = "logreg"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, n_jobs: int = 4):
        # n_jobs=4: each liblinear worker copies the training matrix (~1.2 GB for 73M non-zeros)
        self.C, self.max_iter, self.n_jobs = C, max_iter, n_jobs

    def fit(self, X, Y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        base = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear",
                                  max_iter=self.max_iter)
        self.model_ = OneVsRestClassifier(base, n_jobs=self.n_jobs).fit(X, Y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def coef_(self) -> np.ndarray:
        return np.vstack([e.coef_.ravel() for e in self.model_.estimators_])

    @property
    def intercept_(self) -> np.ndarray:
        return np.array([e.intercept_[0] for e in self.model_.estimators_])


class TreeModel:
    name = "tree"

    def __init__(self, max_depth: Optional[int] = 20, min_samples_leaf: int = 5, seed: int = 42):
        self.max_depth, self.min_samples_leaf, self.seed = max_depth, min_samples_leaf, seed

    def fit(self, X, Y):
        from sklearn.tree import DecisionTreeClassifier

        self.model_ = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                             class_weight="balanced", random_state=self.seed).fit(X, Y)
        return self

    def predict_proba(self, X):
        return _proba_list_to_matrix(self.model_.predict_proba(X))


class ForestModel:
    name = "forest"

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None, min_samples_leaf: int = 2,
                 max_features="sqrt", seed: int = 42, n_jobs: int = -1):
        self.n_estimators, self.max_depth, self.min_samples_leaf = n_estimators, max_depth, min_samples_leaf
        self.max_features, self.seed, self.n_jobs = max_features, seed, n_jobs

    def fit(self, X, Y):
        from sklearn.ensemble import RandomForestClassifier

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, class_weight="balanced_subsample", n_jobs=self.n_jobs,
            random_state=self.seed).fit(X, Y)
        return self

    def predict_proba(self, X):
        return _proba_list_to_matrix(self.model_.predict_proba(X))


# --------------------------------------------------------------------------- #
def l2_normalise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def top_k_cosine(Q: np.ndarray, D: np.ndarray, k: int, chunk: int = 256, exclude_self: bool = False):
    """Top-k cosine neighbours of each row of Q among rows of D (both L2-normalised)."""
    idx = np.empty((Q.shape[0], k), dtype=np.int64)
    sim = np.empty((Q.shape[0], k), dtype=np.float32)
    for s in range(0, Q.shape[0], chunk):
        S = Q[s:s + chunk] @ D.T
        if exclude_self:
            S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        part = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        order = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, 1)
        sim[s:s + chunk] = np.take_along_axis(ps, order, 1)
    return idx, sim


class KNNModel:
    """Cosine KNN on dense (SVD) features; probability = similarity-weighted vote."""

    name = "knn"

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, X, Y):
        self.X_ = l2_normalise(X)
        self.Y_ = np.asarray(Y, dtype=np.float32)
        return self

    def neighbours(self, X, k: Optional[int] = None):
        return top_k_cosine(l2_normalise(X), self.X_, k or self.k)

    def proba_from_neighbours(self, idx, sim, k: Optional[int] = None) -> np.ndarray:
        k = k or self.k
        w = np.clip(sim[:, :k], 1e-6, None)
        return np.einsum("nk,nkc->nc", w, self.Y_[idx[:, :k]]) / w.sum(1, keepdims=True)

    def predict_proba(self, X):
        idx, sim = self.neighbours(X)
        return self.proba_from_neighbours(idx, sim)


# --------------------------------------------------------------------------- #
@dataclass
class MLPHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_pr_auc: List[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0


def _batch_tensor(X, rows, torch, device=None):
    """Rows of X as a float32 tensor on `device`. On a GPU a sparse batch is sent as its non-zeros
    and made dense there, which is much cheaper than densifying on the CPU and copying."""
    xb = X[rows]
    if sp.issparse(xb) and device is not None and device.type == "cuda":
        xb = xb.tocoo()
        idx = torch.from_numpy(np.vstack([xb.row, xb.col]).astype(np.int64)).to(device)
        val = torch.from_numpy(xb.data.astype(np.float32)).to(device)
        return torch.sparse_coo_tensor(idx, val, xb.shape, device=device).to_dense()
    if sp.issparse(xb):
        xb = xb.toarray()
    t = torch.from_numpy(np.asarray(xb, dtype=np.float32))
    return t if device is None else t.to(device)


def _torch_device(torch, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class MLPModel:
    """Two-hidden-layer MLP trained with PyTorch; numpy forward pass for inference."""

    name = "mlp"

    def __init__(self, hidden: Sequence[int] = (512, 128), dropout: float = 0.3, lr: float = 1e-3,
                 weight_decay: float = 1e-5, batch_size: int = 256, max_epochs: int = 30, patience: int = 4,
                 seed: int = 42, max_pos_weight: float = 50.0, verbose: bool = True, device: str = "auto"):
        self.hidden, self.dropout, self.lr, self.weight_decay = list(hidden), dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience, self.seed = batch_size, max_epochs, patience, seed
        self.max_pos_weight, self.verbose, self.device = max_pos_weight, verbose, device
        self.history = MLPHistory()
        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []

    def fit(self, X, Y, X_val=None, Y_val=None):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = _torch_device(torch, getattr(self, "device", "auto"))
        self.device_used_ = str(dev)
        Y = np.asarray(Y, dtype=np.float32)
        d, c = X.shape[1], Y.shape[1]
        layers: List[nn.Module] = []
        prev = d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, c))
        net = nn.Sequential(*layers).to(dev)
        pos = Y.sum(0)
        pos_weight = torch.tensor(np.clip((len(Y) - pos) / np.maximum(pos, 1), 1.0, self.max_pos_weight),
                                  dtype=torch.float32, device=dev)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -np.inf, None, 0
        t0 = time.time()
        for epoch in range(self.max_epochs):
            net.train()
            perm = rng.permutation(X.shape[0])
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                rows = np.sort(perm[s:s + self.batch_size])
                xb = _batch_tensor(X, rows, torch, dev)
                yb = torch.from_numpy(Y[rows]).to(dev)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(rows)
            self.history.train_loss.append(total / X.shape[0])
            if X_val is None:
                continue
            logits = self._torch_logits(net, X_val, torch, dev)
            vl = float(loss_fn(torch.from_numpy(logits).to(dev),
                               torch.from_numpy(np.asarray(Y_val, np.float32)).to(dev)))
            ap = float(np.nanmean(pr_auc(Y_val, 1 / (1 + np.exp(-logits)))))
            self.history.val_loss.append(vl)
            self.history.val_pr_auc.append(ap)
            if self.verbose:
                print(f"      mlp seed={self.seed} epoch {epoch + 1}: train {self.history.train_loss[-1]:.4f}"
                      f" val {vl:.4f} PR-AUC {ap:.4f}", flush=True)
            if ap > best + 1e-4:
                best, bad, self.history.best_epoch = ap, 0, epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.history.seconds = time.time() - t0
        linears = [m for m in net if isinstance(m, nn.Linear)]
        self.weights_ = [m.weight.detach().cpu().numpy().T.astype(np.float32) for m in linears]
        self.biases_ = [m.bias.detach().cpu().numpy().astype(np.float32) for m in linears]
        return self

    def _torch_logits(self, net, X, torch, device=None, batch: int = 2048) -> np.ndarray:
        net.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, X.shape[0], batch):
                rows = np.arange(s, min(s + batch, X.shape[0]))
                outs.append(net(_batch_tensor(X, rows, torch, device)).cpu().numpy())
        return np.vstack(outs)

    # numpy inference (dropout is inactive at inference time)
    def logits(self, X) -> np.ndarray:
        h = X
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            W = W.astype(np.float32, copy=False)
            h = np.asarray(h @ W) + b
            if i < len(self.weights_) - 1:
                h = np.maximum(h, 0.0)
        return np.asarray(h)

    def predict_proba(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(X)))

    def compact(self) -> "MLPModel":
        """Store weights as float16 to halve the file size (computation stays float32)."""
        self.weights_ = [w.astype(np.float16) for w in self.weights_]
        return self
