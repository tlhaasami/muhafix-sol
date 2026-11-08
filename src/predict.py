"""
Muhafiz.sol - prediction API (blueprint 7.4 interface contract, P2.4, P2.6).

    from src.predict import Predictor
    p = Predictor()                      # loads the frozen model-v1 artefacts
    p.predict(source)                    # probabilities, flags, risk_level, latency_ms
    p.similar(source, k=3)               # nearest training contracts

Risk rule (documented in the model card and shown in the app):
  low    - no class reaches its decision threshold
  medium - at least one class reaches its threshold
  high   - at least one class reaches the stricter level t + m * (1 - t)
           (m = ``high_margin`` in thresholds.json)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import load_config, path, read_json
from src.evaluate import high_cutoffs, risk_level
from src.features import FeaturePipeline, NgramAnalyzer
from src.preprocess import normalise_identifiers, prepare_tokens, tokenize_lines, truncate_tokens

MAX_INPUT_CHARS = 1_500_000      # ~ the largest contract in Dataset A
SIMILAR_DIMS = 256               # SVD components used by the similar-contracts index
MIN_TOKENS = 5
SOLIDITY_HINTS = ("contract", "library", "interface", "pragma", "function")


class InputError(ValueError):
    """Raised for input the model cannot score (empty, oversized)."""


@dataclass
class Prepared:
    tokens: List[str]           # tokens the model sees (after the long-file policy)
    n_tokens_full: int
    truncated: bool
    warnings: List[str]


def validate_source(source: str) -> List[str]:
    """Raise InputError for unusable input; return soft warnings otherwise."""
    if source is None or not str(source).strip():
        raise InputError("The input is empty. Paste a Solidity contract or upload a .sol file.")
    if len(source) > MAX_INPUT_CHARS:
        raise InputError(f"The input is too large ({len(source):,} characters; limit {MAX_INPUT_CHARS:,}).")
    warnings = []
    low = source.lower()
    if not any(h in low for h in SOLIDITY_HINTS):
        warnings.append("This does not look like Solidity (no 'contract', 'pragma' or 'function' keyword). "
                        "The scores below are unlikely to be meaningful.")
    return warnings


class Predictor:
    """Loads the frozen artefacts once; thread-safe for read-only use."""

    def __init__(self, models_dir=None, model: Optional[str] = None):
        md = path("models_dir") if models_dir is None else models_dir
        self.cfg = load_config()
        self.label_map = read_json(md / "label_map.json")
        self.meta = read_json(md / "thresholds.json")
        self.classes: List[str] = list(self.meta["classes"])
        self.vec: FeaturePipeline = FeaturePipeline.load(md / "vectorizer.joblib")
        self.models = {fam: joblib.load(md / f"model_{fam}.joblib") for fam in self.meta["models"]}
        self.default_model = model or self.meta["primary"]
        self._sim_index = None
        self._sim_meta = None
        self._md = md
        # analyzer without identifier normalisation: used on tokens that are already normalised
        self._window_counter = None
        pre = self.cfg["preprocessing"]
        self.max_tokens, self.head_fraction = pre["max_tokens"], pre["head_fraction"]

    # ------------------------------------------------------------- helpers --
    @property
    def available_models(self) -> List[str]:
        return list(self.models)

    def thresholds(self, model: Optional[str] = None, scale: float = 1.0) -> np.ndarray:
        """Decision thresholds; ``scale`` < 1 is more sensitive, > 1 stricter (F10)."""
        t = np.asarray(self.meta["models"][model or self.default_model]["thresholds"], dtype=float)
        return np.clip(t * float(scale), 1e-4, 0.9999)

    def prepare(self, source: str) -> Prepared:
        warnings = validate_source(source)
        toks = prepare_tokens(source)
        if len(toks) < MIN_TOKENS:
            raise InputError("The input is too short to analyse (fewer than 5 code tokens after removing comments).")
        kept, truncated = truncate_tokens(toks, self.max_tokens, self.head_fraction)
        if truncated:
            warnings.append(f"The contract is very long ({len(toks):,} tokens). Only the first and last parts "
                            f"({self.max_tokens:,} tokens) were analysed.")
        return Prepared(kept, len(toks), truncated, warnings)

    def _calibrate(self, p: np.ndarray, model: str) -> np.ndarray:
        platt = self.meta["models"][model].get("platt")
        if not platt:
            return p
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        a = np.array([q[0] for q in platt])
        b = np.array([q[1] for q in platt])
        return 1.0 / (1.0 + np.exp(-(z * a + b)))

    def proba_from_tfidf(self, X: sp.spmatrix, model: Optional[str] = None) -> np.ndarray:
        model = model or self.default_model
        m = self.models[model]
        if model in ("knn",) or (model == "mlp" and self.meta["models"]["mlp"]["params"].get("input") == "svd"):
            X = self.vec.transform_svd(X)
        return self._calibrate(np.asarray(m.predict_proba(X)), model)

    def proba_docs(self, docs: Sequence[str], model: Optional[str] = None) -> np.ndarray:
        """Token-string documents -> calibrated probabilities (n, C)."""
        return self.proba_from_tfidf(self.vec.transform(list(docs)), model)

    # ----------------------------------------------------------- public API --
    def predict(self, source: str, model: Optional[str] = None, threshold_scale: float = 1.0) -> Dict:
        """Interface contract 7.4: probabilities, flags, risk level, latency."""
        t0 = time.perf_counter()
        model = model or self.default_model
        prep = self.prepare(source)
        probs = self.proba_docs([" ".join(prep.tokens)], model)[0]
        thr = self.thresholds(model, threshold_scale)
        margin = float(self.meta["high_margin"])
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "flags": {c: bool(p >= t) for c, p, t in zip(self.classes, probs, thr)},
            "thresholds": {c: float(t) for c, t in zip(self.classes, thr)},
            "high_cutoffs": {c: float(h) for c, h in zip(self.classes, high_cutoffs(thr, margin))},
            "risk_level": risk_level(probs, thr, margin),
            "latency_ms": latency,
            "model": model,
            "n_tokens": prep.n_tokens_full,
            "truncated": prep.truncated,
            "warnings": prep.warnings,
        }

    def similar(self, source: str, k: int = 3) -> List[Dict]:
        """Top-k most similar training contracts (cosine on the SVD space)."""
        if self._sim_index is None:
            self._sim_index = np.load(self._md / "similar_index.npy")
            self._sim_meta = pd.read_parquet(self._md / "similar_meta.parquet")
        prep = self.prepare(source)
        q = self.vec.transform_svd(self.vec.transform([" ".join(prep.tokens)]))[0][:self._sim_index.shape[1]]
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = np.empty(len(self._sim_index), dtype=np.float32)
        for s in range(0, len(sims), 20000):
            sims[s:s + 20000] = self._sim_index[s:s + 20000].astype(np.float32) @ q
        top = np.argsort(-sims)[:k]
        out = []
        for i in top:
            r = self._sim_meta.iloc[int(i)]
            out.append({"address": r["address"], "contract_name": r["contract_name"] or "(unnamed)",
                        "labels": list(r["labels"]) or ["no flagged class"],
                        "similarity": float(sims[i]), "snippet": r["snippet"], "n_lines": int(r["n_lines"]),
                        "etherscan_url": f"https://etherscan.io/address/{r['address']}#code"})
        return out

    # ------------------------------------------------ fast occlusion scoring --
    def _counter_plain(self):
        if self._window_counter is None:
            from sklearn.feature_extraction.text import CountVectorizer

            cv = CountVectorizer(analyzer=NgramAnalyzer(self.vec.ngram_range, ids=False),
                                 vocabulary=self.vec.counter.vocabulary_, dtype=np.float32)
            cv.transform([""])
            self._window_counter = cv
        return self._window_counter

    def removal_probabilities(self, tokens: List[str], spans: Sequence[Tuple[int, int]],
                              model: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Probabilities for the full token list and for each copy with tokens[a:b] removed.

        Exact for n-gram TF-IDF: removing a span only changes n-grams that touch it,
        so counts(without S) = counts(full) - counts(L+S+R) + counts(L+R), where L
        and R are the (n-1) tokens around S. Only small windows are vectorised,
        which makes line-level occlusion fast even for long contracts.
        """
        if self.vec.ids:
            tokens = normalise_identifiers(tokens)
        cv = self._counter_plain()
        ctx = max(self.vec.ngram_range) - 1
        full = cv.transform([" ".join(tokens)])
        with_s, without_s = [], []
        for a, b in spans:
            left, right = tokens[max(0, a - ctx):a], tokens[b:b + ctx]
            with_s.append(" ".join(left + tokens[a:b] + right))
            without_s.append(" ".join(left + right))
        if spans:
            counts = sp.vstack([full] * len(spans)) - cv.transform(with_s) + cv.transform(without_s)
            counts.data = np.maximum(counts.data, 0)
            counts.eliminate_zeros()
            allc = sp.vstack([full, counts]).tocsr()
        else:
            allc = full
        X = self.vec.tfidf.transform(allc).astype(np.float32)
        p = self.proba_from_tfidf(X, model)
        return p[0], p[1:]

    def line_tokens(self, source: str) -> Tuple[List[List[str]], List[str], List[int]]:
        """Per-line tokens, the token list the model sees, and each kept token's line."""
        lines = tokenize_lines(source)
        flat, owner = [], []
        for i, toks in enumerate(lines):
            flat.extend(toks)
            owner.extend([i + 1] * len(toks))
        kept, truncated = truncate_tokens(flat, self.max_tokens, self.head_fraction)
        if truncated:
            head = int(self.max_tokens * self.head_fraction)
            tail = self.max_tokens - head
            owner = owner[:head] + owner[len(owner) - tail:]
        return lines, kept, owner


class MockPredictor:
    """Fixed-output stand-in with the exact predict()/similar() schema (P3.1)."""

    classes = ["access-control", "arithmetic", "reentrancy", "unchecked-calls"]
    available_models = ["mock"]
    default_model = "mock"

    def predict(self, source: str, model: Optional[str] = None, threshold_scale: float = 1.0) -> Dict:
        validate_source(source)
        probs = {"access-control": 0.12, "arithmetic": 0.08, "reentrancy": 0.81, "unchecked-calls": 0.44}
        thr = {c: 0.5 * threshold_scale for c in probs}
        return {"probabilities": probs, "flags": {c: probs[c] >= thr[c] for c in probs}, "thresholds": thr,
                "high_cutoffs": {c: 0.7 for c in probs}, "risk_level": "high", "latency_ms": 1.0,
                "model": "mock", "n_tokens": len(source.split()), "truncated": False, "warnings": []}

    def similar(self, source: str, k: int = 3) -> List[Dict]:
        return [{"address": "0x0", "contract_name": "Example", "labels": ["reentrancy"], "similarity": 0.9,
                 "snippet": "contract Example {}", "n_lines": 10, "etherscan_url": "https://etherscan.io"}][:k]
