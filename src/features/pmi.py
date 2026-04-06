from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.features.base import FeatureArtifact, FeatureSet


class PMIVectorizer:
    def __init__(self, **kwargs: Any) -> None:
        self.binary_occurrence = bool(kwargs.pop("binary_occurrence", True))
        self.smoothing = float(kwargs.pop("smoothing", 1e-9))
        self.use_positive_pmi = bool(kwargs.pop("use_positive_pmi", False))
        self.vectorizer = CountVectorizer(**kwargs)
        self.classes_: np.ndarray | None = None
        self.pmi_scores_: np.ndarray | None = None

    def fit(self, texts: list[str], labels: np.ndarray) -> "PMIVectorizer":
        count_matrix = self.vectorizer.fit_transform(texts).tocsr()
        binary_matrix = count_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)

        x_source = binary_matrix if self.binary_occurrence else count_matrix
        classes = np.unique(labels)
        self.classes_ = classes

        label_indicator = np.zeros((x_source.shape[0], len(classes)), dtype=np.float64)
        for class_index, class_value in enumerate(classes):
            label_indicator[:, class_index] = (labels == class_value).astype(np.float64)

        co_occurrence = np.asarray(x_source.T @ label_indicator, dtype=np.float64)
        term_totals = np.asarray(x_source.sum(axis=0)).ravel().astype(np.float64)
        class_totals = label_indicator.sum(axis=0).astype(np.float64)
        total = float(x_source.shape[0])

        numerator = (co_occurrence * total) + self.smoothing
        denominator = np.outer(term_totals, class_totals) + self.smoothing
        pmi = np.log(numerator / denominator)

        if self.use_positive_pmi:
            pmi = np.maximum(pmi, 0.0)

        self.pmi_scores_ = pmi.astype(np.float32)
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        if self.pmi_scores_ is None or self.classes_ is None:
            raise RuntimeError("PMIVectorizer must be fitted before transform().")

        count_matrix = self.vectorizer.transform(texts).tocsr()
        blocks = [
            count_matrix.multiply(self.pmi_scores_[:, class_index].reshape(1, -1))
            for class_index in range(len(self.classes_))
        ]
        return sparse.hstack(blocks, format="csr")

    def fit_transform(self, texts: list[str], labels: np.ndarray) -> sparse.csr_matrix:
        return self.fit(texts, labels).transform(texts)


def build_pmi_features(
    feature_name: str,
    config: dict[str, Any],
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    train_labels: np.ndarray,
) -> FeatureSet:
    vectorizer = PMIVectorizer(
        ngram_range=tuple(config.get("ngram_range", [1, 1])),
        max_features=config.get("max_features"),
        min_df=config.get("min_df", 1),
        max_df=config.get("max_df", 1.0),
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        binary_occurrence=bool(config.get("binary_occurrence", True)),
        smoothing=float(config.get("smoothing", 1e-9)),
        use_positive_pmi=bool(config.get("use_positive_pmi", False)),
    )

    train_x = vectorizer.fit_transform(train_texts, train_labels)
    val_x = vectorizer.transform(val_texts)
    test_x = vectorizer.transform(test_texts)
    vocab_size = len(vectorizer.vectorizer.vocabulary_)
    num_classes = len(vectorizer.classes_) if vectorizer.classes_ is not None else 0
    artifact = FeatureArtifact(
        name=feature_name,
        feature_type="pmi",
        transformer=vectorizer,
        metadata={
            "vocabulary_size": vocab_size,
            "transformed_feature_count": vocab_size * num_classes,
            "config": config,
        },
    )
    return FeatureSet(train_x=train_x, val_x=val_x, test_x=test_x, artifact=artifact)
