from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.features.base import FeatureArtifact, FeatureSet


def _base_vectorizer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ngram_range": tuple(config.get("ngram_range", [1, 1])),
        "max_features": config.get("max_features"),
        "min_df": config.get("min_df", 1),
        "max_df": config.get("max_df", 1.0),
        "tokenizer": str.split,
        "preprocessor": None,
        "token_pattern": None,
        "lowercase": False,
    }


def build_count_features(
    feature_name: str,
    config: dict[str, Any],
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
) -> FeatureSet:
    vectorizer = CountVectorizer(
        **_base_vectorizer_kwargs(config),
        binary=bool(config.get("binary", False)),
    )
    train_x = vectorizer.fit_transform(train_texts)
    val_x = vectorizer.transform(val_texts)
    test_x = vectorizer.transform(test_texts)
    artifact = FeatureArtifact(
        name=feature_name,
        feature_type="count",
        transformer=vectorizer,
        metadata={
            "vocabulary_size": len(vectorizer.vocabulary_),
            "config": config,
        },
    )
    return FeatureSet(train_x=train_x, val_x=val_x, test_x=test_x, artifact=artifact)


def build_tfidf_features(
    feature_name: str,
    config: dict[str, Any],
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
) -> FeatureSet:
    vectorizer = TfidfVectorizer(
        **_base_vectorizer_kwargs(config),
        use_idf=bool(config.get("use_idf", True)),
        smooth_idf=bool(config.get("smooth_idf", True)),
        sublinear_tf=bool(config.get("sublinear_tf", False)),
        norm=config.get("norm", "l2"),
    )
    train_x = vectorizer.fit_transform(train_texts)
    val_x = vectorizer.transform(val_texts)
    test_x = vectorizer.transform(test_texts)
    artifact = FeatureArtifact(
        name=feature_name,
        feature_type="tfidf",
        transformer=vectorizer,
        metadata={
            "vocabulary_size": len(vectorizer.vocabulary_),
            "config": config,
        },
    )
    return FeatureSet(train_x=train_x, val_x=val_x, test_x=test_x, artifact=artifact)
