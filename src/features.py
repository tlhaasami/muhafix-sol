"""
Muhafiz.sol - TF-IDF feature pipeline (blueprint P1.7).

Input documents are *token strings* (tokens produced by src.preprocess joined by
single spaces). The analyzer builds token n-grams (1..3 by default).

Why a custom vocabulary step?  sklearn's ``TfidfVectorizer.fit`` keeps every
distinct n-gram of the corpus in a Python dict before applying
``max_features``; on ~70K flattened contracts with token trigrams that is tens
of millions of strings and does not fit in laptop memory. We therefore:

1. choose the vocabulary on a seeded subset of TRAIN documents only: document
   frequencies are counted per chunk in parallel (n-grams with df < 2 inside a
   chunk are pruned - harmless for the top-k features, whose df is in the
   thousands), merged, filtered by ``min_df``/``max_df`` and cut to the top
   ``max_features`` by document frequency;
2. count every document against that fixed vocabulary in parallel;
3. fit the IDF weights on the full TRAIN count matrix (TfidfTransformer).

Validation, test and SmartBugs documents are only ever *transformed*.
An optional TruncatedSVD (512 dims), also fitted on TRAIN only, gives a dense
representation for KNN and as an MLP alternative.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.preprocess import normalise_identifiers


class NgramAnalyzer:
    """Token string -> list of n-grams. Picklable (needed for joblib and saving)."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3), ids: bool = False):
        self.ngram_range = tuple(ngram_range)
        self.ids = ids

    def __call__(self, doc: str) -> List[str]:
        toks = doc.split()
        if self.ids:
            toks = normalise_identifiers(toks)
        lo, hi = self.ngram_range
        grams: List[str] = []
        for n in range(lo, hi + 1):
            if n == 1:
                grams.extend(toks)
            else:
                grams.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return grams


def _chunk_df(docs: Sequence[str], analyzer: NgramAnalyzer, prune_below: int) -> Counter:
    df: Counter = Counter()
    for d in docs:
        df.update(set(analyzer(d)))
    if prune_below > 1:
        df = Counter({k: v for k, v in df.items() if v >= prune_below})
    return df


def _chunks(seq: Sequence, n: int) -> List[Sequence]:
    size = max(1, int(np.ceil(len(seq) / n)))
    return [seq[i:i + size] for i in range(0, len(seq), size)]


class FeaturePipeline:
    """TF-IDF over Solidity token n-grams with a memory-safe vocabulary step."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3), max_features: int = 20000,
                 min_df: int = 5, max_df: float = 0.95, sublinear_tf: bool = True,
                 ids: bool = False, svd_components: Optional[int] = None,
                 random_state: int = 42, n_jobs: int = -1):
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.ids = ids
        self.svd_components = svd_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.counter: Optional[CountVectorizer] = None
        self.tfidf: Optional[TfidfTransformer] = None
        self.svd = None
        self.vocab_stats: dict = {}

    # ------------------------------------------------------------------ fit --
    def _n_workers(self) -> int:
        import os

        if self.n_jobs in (-1, None):
            return os.cpu_count() or 1
        return max(1, int(self.n_jobs))

    def select_vocabulary(self, docs: Sequence[str]) -> List[str]:
        analyzer = NgramAnalyzer(self.ngram_range, self.ids)
        workers = min(self._n_workers(), max(1, len(docs) // 200))
        if workers > 1:
            parts = joblib.Parallel(n_jobs=workers)(
                joblib.delayed(_chunk_df)(c, analyzer, 2) for c in _chunks(list(docs), workers))
        else:
            parts = [_chunk_df(docs, analyzer, 1)]
        df: Counter = Counter()
        for p in parts:
            df.update(p)
        n = len(docs)
        keep = [(g, c) for g, c in df.items() if c >= self.min_df and c / n <= self.max_df]
        keep.sort(key=lambda x: (-x[1], x[0]))
        vocab = [g for g, _ in keep[:self.max_features]]
        self.vocab_stats = {"vocab_docs": n, "distinct_ngrams_after_chunk_prune": len(df),
                            "eligible_ngrams": len(keep), "vocabulary_size": len(vocab)}
        return vocab

    def fit(self, train_docs: Sequence[str], vocab_docs: Optional[Sequence[str]] = None
            ) -> "FeaturePipeline":
        """Fit on TRAIN documents only. ``vocab_docs`` (a TRAIN subset) selects the vocabulary."""
        vocab = self.select_vocabulary(vocab_docs if vocab_docs is not None else train_docs)
        self.counter = CountVectorizer(analyzer=NgramAnalyzer(self.ngram_range, self.ids),
                                       vocabulary=vocab, dtype=np.float32)
        self.counter.transform([""])  # validates the fixed vocabulary once (sets vocabulary_)
        counts = self.count(train_docs)
        self.tfidf = TfidfTransformer(sublinear_tf=self.sublinear_tf).fit(counts)
        if self.svd_components:
            from sklearn.decomposition import TruncatedSVD

            k = min(self.svd_components, counts.shape[1] - 1, counts.shape[0] - 1)
            self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
            self.svd.fit(self.tfidf.transform(counts))
            self.svd.components_ = self.svd.components_.astype(np.float16)  # quarter of float64 size
        return self

    # ------------------------------------------------------------ transform --
    def count(self, docs: Sequence[str]) -> sp.csr_matrix:
        docs = list(docs)
        workers = min(self._n_workers(), max(1, len(docs) // 500))
        if workers <= 1:
            return self.counter.transform(docs).tocsr()
        parts = joblib.Parallel(n_jobs=workers)(
            joblib.delayed(self.counter.transform)(c) for c in _chunks(docs, workers * 4))
        return sp.vstack(parts).tocsr()

    def transform(self, docs: Iterable[str]) -> sp.csr_matrix:
        if self.counter is None or self.tfidf is None:
            raise ValueError("FeaturePipeline is not fitted yet")
        return self.tfidf.transform(self.count(list(docs))).astype(np.float32).tocsr()

    def transform_svd(self, X: sp.spmatrix) -> np.ndarray:
        if self.svd is None:
            raise ValueError("SVD was not fitted (svd_components=None)")
        # components are stored as float16 on disk (half the size); compute in float32
        comp = self.svd.components_.astype(np.float32, copy=False)
        return np.asarray(X @ comp.T, dtype=np.float32)

    @property
    def vocabulary(self) -> List[str]:
        voc = self.counter.vocabulary_ if hasattr(self.counter, "vocabulary_") else self.counter.vocabulary
        inv = sorted(voc.items(), key=lambda kv: kv[1])
        return [g for g, _ in inv]

    def benchmark_single(self, doc: str, repeats: int = 5) -> float:
        """Median milliseconds to transform one document."""
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.tfidf.transform(self.counter.transform([doc]))
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    # ---------------------------------------------------------- persistence --
    def save(self, p: Path | str) -> None:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=3)

    @staticmethod
    def load(p: Path | str) -> "FeaturePipeline":
        return joblib.load(p)
