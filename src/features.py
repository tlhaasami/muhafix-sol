"""
Muhafiz.sol - TF-IDF Feature Pipeline Module
Phase 1: Work Package P1.7
Owner: Talha Sami (24I-3118)

Implements:
1. Feature extraction using n-gram (1 to 3) TF-IDF fitted strictly on training data.
2. Optional 512-dimensional TruncatedSVD dimensionality reduction for KNN and MLPs.
3. Model serialization using joblib.
4. Single-contract transform benchmark (< 1 second latency).
"""

import os
import time
import joblib
from typing import List, Tuple, Any, Optional
from src.preprocess import clean_source, normalise, tokenize


def solidity_tokenizer(text: str) -> List[str]:
    """Tokenize preprocessed Solidity source."""
    return tokenize(normalise(clean_source(text)))


class FeaturePipeline:
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 3),
        svd_components: Optional[int] = 512,
        random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.svd_components = svd_components
        self.random_state = random_state
        self.vectorizer = None
        self.svd = None

    def fit(self, raw_documents: List[str]):
        """
        Fit vectorizer and optional TruncatedSVD STRICTLY on training documents.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            tokenizer=solidity_tokenizer,
            token_pattern=None,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True
        )
        tfidf_matrix = self.vectorizer.fit_transform(raw_documents)

        if self.svd_components and tfidf_matrix.shape[1] > self.svd_components:
            from sklearn.decomposition import TruncatedSVD
            n_comp = min(self.svd_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
            if n_comp > 1:
                self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
                self.svd.fit(tfidf_matrix)

        return self

    def transform(self, raw_documents: List[str]):
        """Transform documents to sparse TF-IDF feature matrix."""
        if self.vectorizer is None:
            raise ValueError("FeaturePipeline is not fitted yet.")
        return self.vectorizer.transform(raw_documents)

    def transform_svd(self, raw_documents: List[str]):
        """Transform documents to reduced dense representation via TruncatedSVD."""
        sparse_mat = self.transform(raw_documents)
        if self.svd is not None:
            return self.svd.transform(sparse_mat)
        return sparse_mat.toarray()

    def benchmark_single(self, raw_source: str) -> float:
        """Measure inference transformation latency for a single contract."""
        start = time.perf_counter()
        _ = self.transform([raw_source])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return latency_ms

    def save(self, model_path: str = "models/vectorizer.joblib"):
        """Save fitted vectorizer and pipeline."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range
        }, model_path)

    @classmethod
    def load(cls, model_path: str = "models/vectorizer.joblib"):
        """Load fitted pipeline from disk."""
        data = joblib.load(model_path)
        pipe = cls(
            max_features=data.get("max_features", 10000),
            ngram_range=data.get("ngram_range", (1, 3))
        )
        pipe.vectorizer = data["vectorizer"]
        pipe.svd = data.get("svd")
        return pipe
