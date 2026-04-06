#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ml import BernoulliNB, LogisticBinary, MultinomialNB, classification_metrics
from core.paths import SEED
from core.sentiment_task import (
    SKLEARN_AVAILABLE,
    TASK3_CUSTOM_MAX_SAMPLES_DEFAULT,
    _normalize_task3_max_samples,
    _stratified_sample_indices,
    build_vocab_for_classification,
    load_sentiment_dataset,
    sentiment_lexicon_binary_features,
    sentiment_lexicon_features,
    sentiment_lexicon_nonnegative_features,
    vectorize_bow_binary,
    vectorize_bow_counts,
)

if SKLEARN_AVAILABLE:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB as SkBernoulliNB
    from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

MetricValue = float | int | str
FEATURE_SETS = ("bow", "lexicon", "bow_lexicon")
DEFAULT_DATASET_CANDIDATES = (
    ROOT / "sentiment140_100k_clean_balanced_v2.csv",
    ROOT / "sentiment140_100k_clean_balanced.csv",
    ROOT / "sentiment_dataset" / "dataset_v1.csv",
    ROOT / "sentiment_dataset" / "dataset.csv",
)


def default_dataset_path() -> Path:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_DATASET_CANDIDATES[0]


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_weight_list(raw: str) -> List[str]:
    values: List[str] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in {"none", "balanced"}:
            raise ValueError("class weights must be from: none, balanced")
        values.append(token)
    if not values:
        raise ValueError("Expected at least one class-weight value.")
    return values


def parse_reg_type_list(raw: str) -> List[str]:
    values: List[str] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in {"l1", "l2"}:
            raise ValueError("reg types must be from: l1, l2")
        values.append(token)
    if not values:
        raise ValueError("Expected at least one reg type.")
    return values


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_vals: List[float] = []
    for cls in (0, 1):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        f1_vals.append(f1)
    return float(sum(f1_vals) / len(f1_vals))


def _metrics_with_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = classification_metrics(y_true, y_pred)
    out["macro_f1"] = _macro_f1(y_true, y_pred)
    return out


def _sort_key(row: Dict[str, MetricValue], selection_metric: str) -> Tuple[float, float, float, float]:
    dev_acc = float(row["dev_accuracy"])
    dev_macro = float(row["dev_macro_f1"])
    test_acc = float(row["test_accuracy"])
    test_macro = float(row["test_macro_f1"])
    if selection_metric == "dev_accuracy":
        return (dev_acc, dev_macro, test_acc, test_macro)
    return (dev_macro, dev_acc, test_macro, test_acc)


def best_by_metric(
    results: List[Dict[str, MetricValue]],
    selection_metric: str,
) -> Dict[str, MetricValue]:
    return max(results, key=lambda row: _sort_key(row, selection_metric))


def topk_by_metric(
    results: List[Dict[str, MetricValue]],
    selection_metric: str,
    k: int = 5,
) -> List[Dict[str, MetricValue]]:
    rows = sorted(results, key=lambda row: _sort_key(row, selection_metric), reverse=True)
    return rows[: min(k, len(rows))]


def fmt_result(result: Dict[str, MetricValue]) -> str:
    fields = [f"model={result['model']}"]
    if "feature_set" in result:
        fields.append(f"feature_set={result['feature_set']}")
    for key in (
        "alpha",
        "c",
        "class_weight",
        "lr",
        "epochs",
        "reg_type",
        "reg_strength",
    ):
        if key in result:
            fields.append(f"{key}={result[key]}")
    fields.append(f"dev_acc={float(result['dev_accuracy']):.6f}")
    fields.append(f"dev_macro_f1={float(result['dev_macro_f1']):.6f}")
    fields.append(f"test_acc={float(result['test_accuracy']):.6f}")
    fields.append(f"test_macro_f1={float(result['test_macro_f1']):.6f}")
    return " ".join(fields)


def _stratified_split_indices(y: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_ratio)))
        if n_test >= len(cls_idx):
            n_test = max(1, len(cls_idx) - 1)
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def run_sklearn_tuning(
    texts: Sequence[str],
    y: np.ndarray,
    mnb_alphas: List[float],
    bnb_alphas: List[float],
    lr_c_values: List[float],
    class_weights: List[str],
    selection_metric: str,
    min_df: int,
    max_features: int,
    test_ratio: float,
    dev_ratio_within_train: float,
) -> Dict[str, object]:
    x_pool_text, x_test_text, y_pool, y_test = train_test_split(
        list(texts),
        y,
        test_size=test_ratio,
        random_state=SEED,
        stratify=y,
    )
    x_train_text, x_dev_text, y_train, y_dev = train_test_split(
        x_pool_text,
        y_pool,
        test_size=dev_ratio_within_train,
        random_state=SEED + 1,
        stratify=y_pool,
    )

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
    )
    xtr_counts = vectorizer.fit_transform(x_train_text)
    xdv_counts = vectorizer.transform(x_dev_text)
    xte_counts = vectorizer.transform(x_test_text)

    xtr_lex_mnb = sentiment_lexicon_nonnegative_features(x_train_text)
    xdv_lex_mnb = sentiment_lexicon_nonnegative_features(x_dev_text)
    xte_lex_mnb = sentiment_lexicon_nonnegative_features(x_test_text)
    xtr_lex_lr = sentiment_lexicon_features(x_train_text)
    xdv_lex_lr = sentiment_lexicon_features(x_dev_text)
    xte_lex_lr = sentiment_lexicon_features(x_test_text)
    xtr_lex_bin = sentiment_lexicon_binary_features(x_train_text)
    xdv_lex_bin = sentiment_lexicon_binary_features(x_dev_text)
    xte_lex_bin = sentiment_lexicon_binary_features(x_test_text)

    xtr_bow_bin = (xtr_counts > 0).astype(np.float32)
    xdv_bow_bin = (xdv_counts > 0).astype(np.float32)
    xte_bow_bin = (xte_counts > 0).astype(np.float32)
    mnb_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (
            sparse.csr_matrix(xtr_lex_mnb),
            sparse.csr_matrix(xdv_lex_mnb),
            sparse.csr_matrix(xte_lex_mnb),
        ),
        "bow_lexicon": (
            sparse.hstack([xtr_counts, sparse.csr_matrix(xtr_lex_mnb)], format="csr"),
            sparse.hstack([xdv_counts, sparse.csr_matrix(xdv_lex_mnb)], format="csr"),
            sparse.hstack([xte_counts, sparse.csr_matrix(xte_lex_mnb)], format="csr"),
        ),
    }
    bnb_inputs = {
        "bow": (xtr_bow_bin, xdv_bow_bin, xte_bow_bin),
        "lexicon": (
            sparse.csr_matrix(xtr_lex_bin),
            sparse.csr_matrix(xdv_lex_bin),
            sparse.csr_matrix(xte_lex_bin),
        ),
        "bow_lexicon": (
            sparse.hstack([xtr_bow_bin, sparse.csr_matrix(xtr_lex_bin)], format="csr"),
            sparse.hstack([xdv_bow_bin, sparse.csr_matrix(xdv_lex_bin)], format="csr"),
            sparse.hstack([xte_bow_bin, sparse.csr_matrix(xte_lex_bin)], format="csr"),
        ),
    }
    lr_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (
            sparse.csr_matrix(xtr_lex_lr),
            sparse.csr_matrix(xdv_lex_lr),
            sparse.csr_matrix(xte_lex_lr),
        ),
        "bow_lexicon": (
            sparse.hstack([xtr_counts, sparse.csr_matrix(xtr_lex_lr)], format="csr"),
            sparse.hstack([xdv_counts, sparse.csr_matrix(xdv_lex_lr)], format="csr"),
            sparse.hstack([xte_counts, sparse.csr_matrix(xte_lex_lr)], format="csr"),
        ),
    }

    mnb_results: List[Dict[str, MetricValue]] = []
    for feature_set, (xtr, xdv, xte) in mnb_inputs.items():
        for alpha in mnb_alphas:
            model = SkMultinomialNB(alpha=alpha).fit(xtr, y_train)
            pred_dev = model.predict(xdv).astype(np.int64)
            pred_test = model.predict(xte).astype(np.int64)
            dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
            test_metrics = _metrics_with_macro_f1(y_test, pred_test)
            row: Dict[str, MetricValue] = {
                "model": "multinomial_nb",
                "feature_set": feature_set,
                "alpha": float(alpha),
                "dev_accuracy": float(dev_metrics["accuracy"]),
                "dev_f1": float(dev_metrics["f1"]),
                "dev_macro_f1": float(dev_metrics["macro_f1"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1": float(test_metrics["f1"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
            }
            mnb_results.append(row)

    bnb_results: List[Dict[str, MetricValue]] = []
    for feature_set, (xtr, xdv, xte) in bnb_inputs.items():
        for alpha in bnb_alphas:
            model = SkBernoulliNB(alpha=alpha, binarize=0.0).fit(xtr, y_train)
            pred_dev = model.predict(xdv).astype(np.int64)
            pred_test = model.predict(xte).astype(np.int64)
            dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
            test_metrics = _metrics_with_macro_f1(y_test, pred_test)
            row = {
                "model": "bernoulli_nb",
                "feature_set": feature_set,
                "alpha": float(alpha),
                "dev_accuracy": float(dev_metrics["accuracy"]),
                "dev_f1": float(dev_metrics["f1"]),
                "dev_macro_f1": float(dev_metrics["macro_f1"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1": float(test_metrics["f1"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
            }
            bnb_results.append(row)

    lr_results: List[Dict[str, MetricValue]] = []
    for feature_set, (xtr, xdv, xte) in lr_inputs.items():
        for c, class_weight in product(lr_c_values, class_weights):
            model = LogisticRegression(
                C=float(c),
                solver="liblinear",
                max_iter=3000,
                random_state=SEED,
                class_weight=None if class_weight == "none" else "balanced",
            ).fit(xtr, y_train)
            pred_dev = model.predict(xdv).astype(np.int64)
            pred_test = model.predict(xte).astype(np.int64)
            dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
            test_metrics = _metrics_with_macro_f1(y_test, pred_test)
            row = {
                "model": "logistic_regression",
                "feature_set": feature_set,
                "c": float(c),
                "class_weight": class_weight,
                "dev_accuracy": float(dev_metrics["accuracy"]),
                "dev_f1": float(dev_metrics["f1"]),
                "dev_macro_f1": float(dev_metrics["macro_f1"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_f1": float(test_metrics["f1"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
            }
            lr_results.append(row)

    best_mnb = best_by_metric(mnb_results, selection_metric)
    best_bnb = best_by_metric(bnb_results, selection_metric)
    best_lr = best_by_metric(lr_results, selection_metric)
    best_overall = max([best_mnb, best_bnb, best_lr], key=lambda row: _sort_key(row, selection_metric))

    return {
        "backend": "sklearn",
        "split": {
            "train_examples": int(len(y_train)),
            "dev_examples": int(len(y_dev)),
            "test_examples": int(len(y_test)),
        },
        "feature_config": {
            "vectorizer_min_df": int(min_df),
            "vectorizer_max_features": int(max_features),
            "vocab_size": int(len(vectorizer.vocabulary_)),
            "lexicon_feature_count": 6,
            "feature_sets_compared": list(FEATURE_SETS),
        },
        "search": {
            "selection_metric": selection_metric,
            "mnb_alphas": [float(v) for v in mnb_alphas],
            "bnb_alphas": [float(v) for v in bnb_alphas],
            "lr_c_values": [float(v) for v in lr_c_values],
            "class_weights": class_weights,
        },
        "mnb_results": mnb_results,
        "bnb_results": bnb_results,
        "lr_results": lr_results,
        "top_mnb": topk_by_metric(mnb_results, selection_metric),
        "top_bnb": topk_by_metric(bnb_results, selection_metric),
        "top_lr": topk_by_metric(lr_results, selection_metric),
        "best_mnb": best_mnb,
        "best_bnb": best_bnb,
        "best_lr": best_lr,
        "best_overall": best_overall,
    }


def run_custom_tuning(
    texts: Sequence[str],
    y: np.ndarray,
    mnb_alphas: List[float],
    bnb_alphas: List[float],
    lr_values: List[float],
    epoch_values: List[int],
    reg_values: List[float],
    reg_types: List[str],
    selection_metric: str,
    min_vocab_freq: int,
    max_vocab: int,
    test_ratio: float,
    dev_ratio_within_train: float,
) -> Dict[str, object]:
    train_pool_idx, test_idx = _stratified_split_indices(y, test_ratio=test_ratio, seed=SEED)
    rel_train_idx, rel_dev_idx = _stratified_split_indices(
        y[train_pool_idx],
        test_ratio=dev_ratio_within_train,
        seed=SEED + 1,
    )
    train_idx = train_pool_idx[rel_train_idx]
    dev_idx = train_pool_idx[rel_dev_idx]

    x_train_text = [texts[i] for i in train_idx]
    x_dev_text = [texts[i] for i in dev_idx]
    x_test_text = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_dev = y[dev_idx]
    y_test = y[test_idx]

    vocab = build_vocab_for_classification(
        x_train_text,
        min_freq=min_vocab_freq,
        max_vocab=max_vocab,
    )
    xtr_counts = vectorize_bow_counts(x_train_text, vocab)
    xdv_counts = vectorize_bow_counts(x_dev_text, vocab)
    xte_counts = vectorize_bow_counts(x_test_text, vocab)
    xtr_binary = vectorize_bow_binary(x_train_text, vocab)
    xdv_binary = vectorize_bow_binary(x_dev_text, vocab)
    xte_binary = vectorize_bow_binary(x_test_text, vocab)

    xtr_lex_mnb = sentiment_lexicon_nonnegative_features(x_train_text)
    xdv_lex_mnb = sentiment_lexicon_nonnegative_features(x_dev_text)
    xte_lex_mnb = sentiment_lexicon_nonnegative_features(x_test_text)
    xtr_lex_lr = sentiment_lexicon_features(x_train_text)
    xdv_lex_lr = sentiment_lexicon_features(x_dev_text)
    xte_lex_lr = sentiment_lexicon_features(x_test_text)
    xtr_lex_bin = sentiment_lexicon_binary_features(x_train_text)
    xdv_lex_bin = sentiment_lexicon_binary_features(x_dev_text)
    xte_lex_bin = sentiment_lexicon_binary_features(x_test_text)

    mnb_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (xtr_lex_mnb, xdv_lex_mnb, xte_lex_mnb),
        "bow_lexicon": (
            np.hstack([xtr_counts, xtr_lex_mnb]),
            np.hstack([xdv_counts, xdv_lex_mnb]),
            np.hstack([xte_counts, xte_lex_mnb]),
        ),
    }
    bnb_inputs = {
        "bow": (xtr_binary, xdv_binary, xte_binary),
        "lexicon": (xtr_lex_bin, xdv_lex_bin, xte_lex_bin),
        "bow_lexicon": (
            np.hstack([xtr_binary, xtr_lex_bin]),
            np.hstack([xdv_binary, xdv_lex_bin]),
            np.hstack([xte_binary, xte_lex_bin]),
        ),
    }
    lr_inputs = {
        "bow": (xtr_counts, xdv_counts, xte_counts),
        "lexicon": (xtr_lex_lr, xdv_lex_lr, xte_lex_lr),
        "bow_lexicon": (
            np.hstack([xtr_counts, xtr_lex_lr]),
            np.hstack([xdv_counts, xdv_lex_lr]),
            np.hstack([xte_counts, xte_lex_lr]),
        ),
    }

    mnb_results: List[Dict[str, MetricValue]] = []
    for feature_set, (xtr, xdv, xte) in mnb_inputs.items():
        for alpha in mnb_alphas:
            model = MultinomialNB(alpha=alpha).fit(xtr, y_train)
            pred_dev = model.predict(xdv)
            pred_test = model.predict(xte)
            dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
            test_metrics = _metrics_with_macro_f1(y_test, pred_test)
            mnb_results.append(
                {
                    "model": "multinomial_nb",
                    "feature_set": feature_set,
                    "alpha": float(alpha),
                    "dev_accuracy": float(dev_metrics["accuracy"]),
                    "dev_f1": float(dev_metrics["f1"]),
                    "dev_macro_f1": float(dev_metrics["macro_f1"]),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_f1": float(test_metrics["f1"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                }
            )

    bnb_results: List[Dict[str, MetricValue]] = []
    for feature_set, (xtr, xdv, xte) in bnb_inputs.items():
        for alpha in bnb_alphas:
            model = BernoulliNB(alpha=alpha).fit(xtr, y_train)
            pred_dev = model.predict(xdv)
            pred_test = model.predict(xte)
            dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
            test_metrics = _metrics_with_macro_f1(y_test, pred_test)
            bnb_results.append(
                {
                    "model": "bernoulli_nb",
                    "feature_set": feature_set,
                    "alpha": float(alpha),
                    "dev_accuracy": float(dev_metrics["accuracy"]),
                    "dev_f1": float(dev_metrics["f1"]),
                    "dev_macro_f1": float(dev_metrics["macro_f1"]),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_f1": float(test_metrics["f1"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                }
            )

    lr_results: List[Dict[str, MetricValue]] = []
    for feature_set, (xtr, xdv, xte) in lr_inputs.items():
        for reg_type, lr, epochs, reg_strength in product(
            reg_types,
            lr_values,
            epoch_values,
            reg_values,
        ):
            model = LogisticBinary(
                lr=float(lr),
                epochs=int(epochs),
                reg_type=reg_type,
                reg_strength=float(reg_strength),
            ).fit(xtr, y_train)
            pred_dev = model.predict(xdv)
            pred_test = model.predict(xte)
            dev_metrics = _metrics_with_macro_f1(y_dev, pred_dev)
            test_metrics = _metrics_with_macro_f1(y_test, pred_test)
            lr_results.append(
                {
                    "model": "logistic_regression_custom",
                    "feature_set": feature_set,
                    "reg_type": reg_type,
                    "lr": float(lr),
                    "epochs": int(epochs),
                    "reg_strength": float(reg_strength),
                    "dev_accuracy": float(dev_metrics["accuracy"]),
                    "dev_f1": float(dev_metrics["f1"]),
                    "dev_macro_f1": float(dev_metrics["macro_f1"]),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_f1": float(test_metrics["f1"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                }
            )

    best_mnb = best_by_metric(mnb_results, selection_metric)
    best_bnb = best_by_metric(bnb_results, selection_metric)
    best_lr = best_by_metric(lr_results, selection_metric)
    best_overall = max([best_mnb, best_bnb, best_lr], key=lambda row: _sort_key(row, selection_metric))

    return {
        "backend": "custom",
        "split": {
            "train_examples": int(len(y_train)),
            "dev_examples": int(len(y_dev)),
            "test_examples": int(len(y_test)),
        },
        "feature_config": {
            "min_vocab_freq": int(min_vocab_freq),
            "max_vocab": int(max_vocab),
            "vocab_size": int(len(vocab)),
            "lexicon_feature_count": 6,
            "feature_sets_compared": list(FEATURE_SETS),
        },
        "search": {
            "selection_metric": selection_metric,
            "mnb_alphas": [float(v) for v in mnb_alphas],
            "bnb_alphas": [float(v) for v in bnb_alphas],
            "lr_values": [float(v) for v in lr_values],
            "epoch_values": [int(v) for v in epoch_values],
            "reg_values": [float(v) for v in reg_values],
            "reg_types": reg_types,
        },
        "mnb_results": mnb_results,
        "bnb_results": bnb_results,
        "lr_results": lr_results,
        "top_mnb": topk_by_metric(mnb_results, selection_metric),
        "top_bnb": topk_by_metric(bnb_results, selection_metric),
        "top_lr": topk_by_metric(lr_results, selection_metric),
        "best_mnb": best_mnb,
        "best_bnb": best_bnb,
        "best_lr": best_lr,
        "best_overall": best_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune Task 1 sentiment models using the cleaned balanced Sentiment140 dataset by default."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset_path(),
        help=(
            "Sentiment CSV path. Defaults to the root-level "
            "`sentiment140_100k_clean_balanced_v2.csv` dataset."
        ),
    )
    parser.add_argument("--search-mode", choices=["quick", "extended"], default="extended")
    parser.add_argument(
        "--selection-metric",
        choices=["dev_macro_f1", "dev_accuracy"],
        default="dev_macro_f1",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--dev-ratio-within-train", type=float, default=0.2)
    parser.add_argument("--mnb-alphas", type=str, default=None)
    parser.add_argument("--bnb-alphas", type=str, default=None)
    parser.add_argument("--lr-c-values", type=str, default=None)
    parser.add_argument("--class-weights", type=str, default=None)
    parser.add_argument("--custom-lr-values", type=str, default=None)
    parser.add_argument("--custom-epoch-values", type=str, default=None)
    parser.add_argument("--custom-reg-values", type=str, default=None)
    parser.add_argument("--custom-reg-types", type=str, default=None)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-vocab-freq", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Cap total tuning dataset size before train/dev/test split. "
            f"Default is {TASK3_CUSTOM_MAX_SAMPLES_DEFAULT} when sklearn is unavailable; "
            "otherwise uses full dataset. Set <=0 to disable cap."
        ),
    )
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    texts, labels, data_source = load_sentiment_dataset(args.dataset_path)
    original_num_samples = len(texts)
    effective_max_samples = _normalize_task3_max_samples(args.max_samples)

    if len(texts) < 500:
        raise RuntimeError(f"Not enough sentiment samples in dataset: {len(texts)} ({data_source})")

    y = np.array(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Dataset must contain both positive and negative classes.")

    sampled_for_memory = 0.0
    if effective_max_samples is not None and len(texts) > effective_max_samples:
        sample_idx = _stratified_sample_indices(y, effective_max_samples)
        texts = [texts[int(i)] for i in sample_idx.tolist()]
        y = y[sample_idx]
        sampled_for_memory = 1.0

    if args.search_mode == "quick":
        default_mnb_alphas = [0.1, 0.5, 1.0]
        default_bnb_alphas = [0.1, 0.5, 1.0]
        default_lr_c_values = [0.3, 1.0, 3.0]
        default_class_weights = ["none", "balanced"]
        default_custom_lr_values = [0.05, 0.1, 0.2]
        default_custom_epoch_values = [25, 35]
        default_custom_reg_values = [1e-5, 1e-4, 1e-3]
        default_custom_reg_types = ["l2"]
    else:
        default_mnb_alphas = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
        default_bnb_alphas = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
        default_lr_c_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
        default_class_weights = ["none", "balanced"]
        default_custom_lr_values = [0.03, 0.05, 0.1, 0.2]
        default_custom_epoch_values = [25, 35, 50]
        default_custom_reg_values = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        default_custom_reg_types = ["l2", "l1"]

    mnb_alphas = parse_float_list(args.mnb_alphas) if args.mnb_alphas else default_mnb_alphas
    bnb_alphas = parse_float_list(args.bnb_alphas) if args.bnb_alphas else default_bnb_alphas
    lr_c_values = parse_float_list(args.lr_c_values) if args.lr_c_values else default_lr_c_values
    class_weights = (
        parse_weight_list(args.class_weights) if args.class_weights else default_class_weights
    )
    custom_lr_values = (
        parse_float_list(args.custom_lr_values)
        if args.custom_lr_values
        else default_custom_lr_values
    )
    custom_epoch_values = (
        parse_int_list(args.custom_epoch_values)
        if args.custom_epoch_values
        else default_custom_epoch_values
    )
    custom_reg_values = (
        parse_float_list(args.custom_reg_values)
        if args.custom_reg_values
        else default_custom_reg_values
    )
    custom_reg_types = (
        parse_reg_type_list(args.custom_reg_types)
        if args.custom_reg_types
        else default_custom_reg_types
    )

    print(f"dataset={args.dataset_path}")
    print(f"data_source={data_source}")
    print(
        f"samples={len(texts)} "
        f"(original={original_num_samples}, "
        f"sampled_for_memory={int(sampled_for_memory)}, "
        f"max_samples={effective_max_samples if effective_max_samples is not None else 'none'}) "
        f"positive_ratio={float(y.mean()):.6f}"
    )
    print(f"backend={'sklearn' if SKLEARN_AVAILABLE else 'custom'}")
    print(f"search_mode={args.search_mode} selection_metric={args.selection_metric}")

    if SKLEARN_AVAILABLE:
        payload = run_sklearn_tuning(
            texts=texts,
            y=y,
            mnb_alphas=mnb_alphas,
            bnb_alphas=bnb_alphas,
            lr_c_values=lr_c_values,
            class_weights=class_weights,
            selection_metric=args.selection_metric,
            min_df=args.min_df,
            max_features=args.max_features,
            test_ratio=args.test_ratio,
            dev_ratio_within_train=args.dev_ratio_within_train,
        )
    else:
        payload = run_custom_tuning(
            texts=texts,
            y=y,
            mnb_alphas=mnb_alphas,
            bnb_alphas=bnb_alphas,
            lr_values=custom_lr_values,
            epoch_values=custom_epoch_values,
            reg_values=custom_reg_values,
            reg_types=custom_reg_types,
            selection_metric=args.selection_metric,
            min_vocab_freq=args.min_vocab_freq,
            max_vocab=args.max_vocab,
            test_ratio=args.test_ratio,
            dev_ratio_within_train=args.dev_ratio_within_train,
        )

    payload["dataset_path"] = str(args.dataset_path)
    payload["data_source"] = data_source
    payload["num_samples"] = int(len(texts))
    payload["num_samples_original"] = int(original_num_samples)
    payload["sampled_for_memory"] = sampled_for_memory
    payload["task3_max_samples"] = (
        float(effective_max_samples) if effective_max_samples is not None else -1.0
    )

    print("\nBest configs")
    print(f"MNB: {fmt_result(payload['best_mnb'])}")
    print(f"BNB: {fmt_result(payload['best_bnb'])}")
    print(f"LR : {fmt_result(payload['best_lr'])}")
    print("\nFINAL_BEST_CONFIG")
    print(fmt_result(payload["best_overall"]))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved tuning report: {args.save_json}")


if __name__ == "__main__":
    main()
