from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics
from src.features.base import FeatureArtifact, FeatureSet
from src.preprocessing.text_cleaner import TextPreprocessor
from src.training.data_loading import DatasetBundle, load_csv_dataset
from src.training.splits import SplitIndices, resolve_split_indices
from src.utils.config import get_required_mapping, load_yaml_config
from src.utils.filesystem import ensure_directories, make_run_id, resolve_path, save_json, save_yaml, slugify
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_global_seed


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "main_sentiment140_full.yaml"
VALID_SEARCH_PROFILES = {"quick", "balanced", "full"}
VALID_SELECTION_METRICS = {"accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1"}
VALID_PARALLEL_BACKENDS = {"threads", "processes"}


@dataclass
class PreparedDataset:
    dataset: DatasetBundle
    processed_texts: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray


@dataclass
class DatasetView:
    dataset_name: str
    preprocessing_name: str
    dataset: DatasetBundle
    split_indices: SplitIndices
    train_texts: np.ndarray
    val_texts: np.ndarray
    test_texts: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
    test_row_ids: np.ndarray


@dataclass
class BaseFeatureBundle:
    feature_name: str
    base_config: dict[str, Any]
    vectorizer: CountVectorizer
    train_counts: sparse.csr_matrix
    val_counts: sparse.csr_matrix
    test_counts: sparse.csr_matrix
    build_seconds: float


@dataclass
class TrialResult:
    experiment_name: str
    dataset_name: str
    preprocessing_name: str
    search_stage: str
    model_name: str
    model_type: str
    vectorizer_config: dict[str, Any]
    model_config: dict[str, Any]
    feature_dimension: int
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    train_seconds: float
    inference_seconds: float
    selection_metric: str
    selection_score: float
    warning_messages: list[str]

    def to_summary_row(self, run_id: str, view: DatasetView, preprocessing_config: dict[str, Any]) -> dict[str, Any]:
        vectorizer_type = str(self.vectorizer_config["type"])
        ngram_range = tuple(int(value) for value in self.vectorizer_config["ngram_range"])
        return {
            "run_id": run_id,
            "experiment_name": self.experiment_name,
            "dataset": self.dataset_name,
            "preprocessing": self.preprocessing_name,
            "search_stage": self.search_stage,
            "model": self.model_name,
            "model_type": self.model_type,
            "dataset_path": str(view.dataset.path),
            "split_artifact": str(view.split_indices.artifact_path),
            "vectorizer_type": vectorizer_type,
            "feature_type": vectorizer_type,
            "ngram_range": f"({ngram_range[0]}, {ngram_range[1]})",
            "max_features": self.vectorizer_config.get("max_features"),
            "min_df": self.vectorizer_config.get("min_df"),
            "max_df": self.vectorizer_config.get("max_df"),
            "preprocessing_lowercase": bool(preprocessing_config.get("lowercase", True)),
            "vectorizer_lowercase": bool(self.vectorizer_config.get("lowercase", False)),
            "binary": self.vectorizer_config.get("binary"),
            "sublinear_tf": self.vectorizer_config.get("sublinear_tf"),
            "feature_dimension": int(self.feature_dimension),
            "train_rows": len(view.split_indices.train_indices),
            "val_rows": len(view.split_indices.val_indices),
            "test_rows": len(view.split_indices.test_indices),
            "accuracy": self.test_metrics["accuracy"],
            "precision": self.test_metrics["precision"],
            "recall": self.test_metrics["recall"],
            "f1": self.test_metrics["f1"],
            "macro_f1": self.test_metrics["macro_f1"],
            "weighted_f1": self.test_metrics["weighted_f1"],
            "val_accuracy": self.val_metrics["accuracy"],
            "val_precision": self.val_metrics["precision"],
            "val_recall": self.val_metrics["recall"],
            "val_f1": self.val_metrics["f1"],
            "val_macro_f1": self.val_metrics["macro_f1"],
            "val_weighted_f1": self.val_metrics["weighted_f1"],
            "train_accuracy": self.train_metrics["accuracy"],
            "train_precision": self.train_metrics["precision"],
            "train_recall": self.train_metrics["recall"],
            "train_f1": self.train_metrics["f1"],
            "train_macro_f1": self.train_metrics["macro_f1"],
            "confusion_matrix_json": json.dumps(self.test_metrics["confusion_matrix"]),
            "val_confusion_matrix_json": json.dumps(self.val_metrics["confusion_matrix"]),
            "train_confusion_matrix_json": json.dumps(self.train_metrics["confusion_matrix"]),
            "selection_metric": self.selection_metric,
            "selection_score": self.selection_score,
            "training_time_seconds": self.train_seconds,
            "inference_time_seconds": self.inference_seconds,
            "warning_count": len(self.warning_messages),
            "warnings_json": json.dumps(self.warning_messages),
            "vectorizer_config_json": json.dumps(_normalize_for_serialization(self.vectorizer_config), sort_keys=True),
            "model_config_json": json.dumps(_normalize_for_serialization(self.model_config), sort_keys=True),
            "lr_c": self.model_config.get("c"),
            "lr_penalty": self.model_config.get("penalty"),
            "lr_solver": self.model_config.get("solver"),
            "lr_class_weight": self.model_config.get("class_weight"),
            "lr_max_iter": self.model_config.get("max_iter"),
            "nb_alpha": self.model_config.get("alpha"),
            "nb_fit_prior": self.model_config.get("fit_prior"),
        }


@dataclass
class TrialOutcome:
    result: TrialResult | None
    error_message: str | None
    model_config: dict[str, Any]


def _normalize_for_serialization(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_for_serialization(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_serialization(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_for_serialization(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _vectorizer_signature(config: dict[str, Any]) -> str:
    return json.dumps(_normalize_for_serialization(config), sort_keys=True)


def _model_signature(config: dict[str, Any]) -> str:
    return json.dumps(_normalize_for_serialization(config), sort_keys=True)


def _base_vectorizer_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ngram_range": list(config.get("ngram_range", [1, 1])),
        "max_features": config.get("max_features"),
        "min_df": int(config.get("min_df", 1)),
        "max_df": float(config.get("max_df", 1.0)),
        "lowercase": bool(config.get("lowercase", False)),
    }


def _base_vectorizer_signature(config: dict[str, Any]) -> str:
    return json.dumps(_normalize_for_serialization(_base_vectorizer_config(config)), sort_keys=True)


def _base_vectorizer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "ngram_range": tuple(config.get("ngram_range", [1, 1])),
        "max_features": config.get("max_features"),
        "min_df": int(config.get("min_df", 1)),
        "max_df": float(config.get("max_df", 1.0)),
        "tokenizer": str.split,
        "preprocessor": None,
        "token_pattern": None,
        "lowercase": bool(config.get("lowercase", False)),
        "dtype": np.float64,
    }


def _sorted_csr(matrix: Any) -> sparse.csr_matrix:
    csr = sparse.csr_matrix(matrix)
    csr.sort_indices()
    return csr


def _binary_sparse_copy(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    binary = matrix.copy()
    if binary.nnz:
        binary.data[:] = 1.0
    binary.sort_indices()
    return binary


def _validate_split_indices(dataset: DatasetBundle, split_indices: SplitIndices) -> None:
    train_indices = np.asarray(split_indices.train_indices, dtype=np.int64)
    val_indices = np.asarray(split_indices.val_indices, dtype=np.int64)
    test_indices = np.asarray(split_indices.test_indices, dtype=np.int64)

    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Train and validation splits overlap.")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Train and test splits overlap.")
    if np.intersect1d(val_indices, test_indices).size:
        raise ValueError("Validation and test splits overlap.")

    unique_count = np.unique(np.concatenate([train_indices, val_indices, test_indices])).size
    if unique_count != len(train_indices) + len(val_indices) + len(test_indices):
        raise ValueError("Split indices contain duplicates.")
    if unique_count != len(dataset.row_ids):
        raise ValueError(
            f"Split coverage mismatch for dataset '{dataset.name}': expected {len(dataset.row_ids)} rows, got {unique_count}."
        )


def _selection_key(result: TrialResult) -> tuple[float, float, float, float]:
    return (
        float(result.selection_score),
        float(result.val_metrics.get("f1", 0.0)),
        float(result.val_metrics.get("accuracy", 0.0)),
        -float(len(result.warning_messages)),
    )


class TraditionalMLTuner:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        dataset_filters: list[str] | None = None,
        preprocessing_filters: list[str] | None = None,
        search_profile: str = "balanced",
        selection_metric: str = "f1",
        top_k_vectorizers: int | None = None,
    ) -> None:
        if search_profile not in VALID_SEARCH_PROFILES:
            raise ValueError(f"Unsupported search profile: {search_profile}")
        if selection_metric not in VALID_SELECTION_METRICS:
            raise ValueError(f"Unsupported selection metric: {selection_metric}")

        self.config = config
        self.config_path = config_path
        self.config_dir = config_path.parent
        self.dataset_filters = set(dataset_filters or [])
        self.preprocessing_filters = set(preprocessing_filters or [])
        self.search_profile = search_profile
        self.selection_metric = selection_metric

        self.datasets_config = get_required_mapping(config, "datasets")
        self.preprocessing_config = get_required_mapping(config, "preprocessing")
        self.output_config = config.get("output", {})
        self.runtime_config = config.get("runtime", {})
        self.tuning_config = config.get("traditional_ml_tuning", {})
        if self.tuning_config and not isinstance(self.tuning_config, dict):
            raise ValueError("Config key 'traditional_ml_tuning' must be a mapping when provided.")

        self.seed = int(config.get("seed", 42))

        run_name = str(self.output_config.get("run_name", self.config_path.stem))
        self.run_id = make_run_id(f"{run_name}_traditional_ml_tuning")

        self.results_root = resolve_path(self.output_config.get("results_dir", "results"), config_dir=self.config_dir)
        self.logs_root = resolve_path(self.output_config.get("logs_dir", "logs"), config_dir=self.config_dir)
        self.models_root = resolve_path(self.output_config.get("models_dir", "models"), config_dir=self.config_dir)
        self.splits_root = resolve_path(self.output_config.get("splits_dir", "splits"), config_dir=self.config_dir)
        self.configs_root = resolve_path(self.output_config.get("configs_dir", "configs"), config_dir=self.config_dir)

        ensure_directories([self.results_root, self.logs_root, self.models_root, self.splits_root, self.configs_root])

        self.results_run_dir = self.results_root / self.run_id
        self.models_run_dir = self.models_root / self.run_id
        self.feature_artifact_dir = self.models_run_dir / "feature_artifacts"
        self.model_artifact_dir = self.models_run_dir / "model_artifacts"
        self.experiment_detail_dir = self.results_run_dir / "experiment_details"
        ensure_directories(
            [
                self.results_run_dir,
                self.models_run_dir,
                self.feature_artifact_dir,
                self.model_artifact_dir,
                self.experiment_detail_dir,
            ]
        )

        self.log_file = self.logs_root / f"{self.run_id}.log"
        self.logger = setup_logger(self.log_file)

        self.datasets_cache: dict[str, DatasetBundle] = {}
        self.splits_cache: dict[str, SplitIndices] = {}
        self.preprocessed_cache: dict[tuple[str, str], PreparedDataset] = {}
        self.dataset_view_cache: dict[tuple[str, str], DatasetView] = {}
        self.base_feature_cache: dict[tuple[str, str, str], BaseFeatureBundle] = {}
        self.feature_cache: dict[tuple[str, str, str], FeatureSet] = {}
        self.profile_records: list[dict[str, Any]] = []
        self.cache_stats: defaultdict[str, int] = defaultdict(int)

        profile_defaults = self._profile_defaults(search_profile)
        configured_top_k = self.tuning_config.get("top_k_vectorizers")
        resolved_top_k = top_k_vectorizers if top_k_vectorizers is not None else configured_top_k
        self.top_k_vectorizers = int(resolved_top_k if resolved_top_k is not None else profile_defaults["top_k_vectorizers"])
        if self.top_k_vectorizers < 1:
            raise ValueError("top_k_vectorizers must be at least 1.")

        self.n_jobs = int(self.tuning_config.get("n_jobs", -1))
        self.parallel_backend = str(self.tuning_config.get("parallel_backend", "threads")).lower()
        if self.parallel_backend not in VALID_PARALLEL_BACKENDS:
            raise ValueError(
                f"traditional_ml_tuning.parallel_backend must be one of {sorted(VALID_PARALLEL_BACKENDS)}, "
                f"got '{self.parallel_backend}'."
            )
        self.parallel_trial_threshold = int(self.tuning_config.get("parallel_trial_threshold", 4))
        self.pre_dispatch = str(self.tuning_config.get("pre_dispatch", "n_jobs"))

    def _record_profile(self, stage: str, seconds: float, **details: Any) -> None:
        payload = {
            "stage": stage,
            "seconds": float(seconds),
            **{str(key): _normalize_for_serialization(value) for key, value in details.items()},
        }
        self.profile_records.append(payload)
        detail_text = " ".join(f"{key}={value}" for key, value in payload.items() if key not in {"stage", "seconds"})
        if detail_text:
            self.logger.info("PROFILE stage=%s seconds=%.3f %s", stage, seconds, detail_text)
        else:
            self.logger.info("PROFILE stage=%s seconds=%.3f", stage, seconds)

    def _profile_defaults(self, profile: str) -> dict[str, Any]:
        if profile == "quick":
            return {
                "ngram_ranges": [(1, 1), (1, 2)],
                "max_features": [5000, 10000],
                "min_df": [1, 2],
                "max_df": [0.95, 1.0],
                "top_k_vectorizers": 2,
            }
        if profile == "full":
            return {
                "ngram_ranges": [(1, 1), (1, 2), (1, 3)],
                "max_features": [None, 5000, 10000, 20000, 30000],
                "min_df": [1, 2, 3, 5],
                "max_df": [0.95, 0.98, 1.0],
                "top_k_vectorizers": 8,
            }
        return {
            "ngram_ranges": [(1, 1), (1, 2), (1, 3)],
            "max_features": [None, 5000, 10000, 20000, 30000],
            "min_df": [1, 2, 3, 5],
            "max_df": [0.95, 0.98, 1.0],
            "top_k_vectorizers": 5,
        }

    def _selected_names(self, config_mapping: dict[str, Any], filters: set[str], kind: str) -> list[str]:
        available = list(config_mapping.keys())
        if not filters:
            return available

        missing = sorted(filters.difference(config_mapping.keys()))
        if missing:
            raise ValueError(f"Unknown {kind}: {missing}")
        return [name for name in available if name in filters]

    def _get_dataset(self, dataset_name: str) -> DatasetBundle:
        if dataset_name in self.datasets_cache:
            self.cache_stats["dataset_hit"] += 1
            return self.datasets_cache[dataset_name]

        self.cache_stats["dataset_miss"] += 1
        dataset_config = self.datasets_config[dataset_name]
        start = time.perf_counter()
        dataset = load_csv_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            config_dir=self.config_dir,
            seed=self.seed,
        )
        elapsed = time.perf_counter() - start
        self.datasets_cache[dataset_name] = dataset
        self.logger.info("Loaded dataset '%s' with %d rows from %s", dataset_name, len(dataset.texts), dataset.path)
        self._record_profile("load_dataset", elapsed, dataset_name=dataset_name, rows=len(dataset.texts), path=dataset.path)
        return dataset

    def _get_split_indices(self, dataset_name: str) -> SplitIndices:
        if dataset_name in self.splits_cache:
            self.cache_stats["split_hit"] += 1
            return self.splits_cache[dataset_name]

        self.cache_stats["split_miss"] += 1
        dataset = self._get_dataset(dataset_name)
        dataset_config = self.datasets_config[dataset_name]
        split_config = dataset_config.get("split")
        if not isinstance(split_config, dict) or "artifact_name" not in split_config:
            raise ValueError(f"Dataset '{dataset_name}' must define a split config with 'artifact_name'.")

        start = time.perf_counter()
        split_indices = resolve_split_indices(dataset=dataset, split_config=split_config, splits_dir=self.splits_root)
        _validate_split_indices(dataset=dataset, split_indices=split_indices)
        elapsed = time.perf_counter() - start

        self.splits_cache[dataset_name] = split_indices
        self.logger.info(
            "Resolved split '%s' for dataset '%s' -> train=%d val=%d test=%d",
            split_indices.artifact_name,
            dataset_name,
            len(split_indices.train_indices),
            len(split_indices.val_indices),
            len(split_indices.test_indices),
        )
        self._record_profile(
            "resolve_split_indices",
            elapsed,
            dataset_name=dataset_name,
            artifact_name=split_indices.artifact_name,
            train_rows=len(split_indices.train_indices),
            val_rows=len(split_indices.val_indices),
            test_rows=len(split_indices.test_indices),
        )
        return split_indices

    def _get_prepared_dataset(self, dataset_name: str, preprocessing_name: str) -> PreparedDataset:
        cache_key = (dataset_name, preprocessing_name)
        if cache_key in self.preprocessed_cache:
            self.cache_stats["preprocess_hit"] += 1
            return self.preprocessed_cache[cache_key]

        self.cache_stats["preprocess_miss"] += 1
        dataset = self._get_dataset(dataset_name)
        preprocessing_cfg = self.preprocessing_config[preprocessing_name]
        preprocessor = TextPreprocessor(config=preprocessing_cfg, config_dir=self.config_dir)

        start = time.perf_counter()
        processed_texts, _token_lists = preprocessor.preprocess_many(dataset.texts)
        elapsed = time.perf_counter() - start

        prepared = PreparedDataset(
            dataset=dataset,
            processed_texts=np.asarray(processed_texts, dtype=object),
            labels=np.asarray(dataset.labels, dtype=np.int64),
            row_ids=np.asarray(dataset.row_ids, dtype=object),
        )
        self.preprocessed_cache[cache_key] = prepared
        self.logger.info(
            "Prepared dataset '%s' using preprocessing '%s'",
            dataset_name,
            preprocessing_name,
        )
        self._record_profile(
            "preprocess_dataset",
            elapsed,
            dataset_name=dataset_name,
            preprocessing_name=preprocessing_name,
            rows=len(processed_texts),
        )
        return prepared

    def _build_dataset_view(self, dataset_name: str, preprocessing_name: str) -> DatasetView:
        cache_key = (dataset_name, preprocessing_name)
        if cache_key in self.dataset_view_cache:
            self.cache_stats["view_hit"] += 1
            return self.dataset_view_cache[cache_key]

        self.cache_stats["view_miss"] += 1
        dataset = self._get_dataset(dataset_name)
        split_indices = self._get_split_indices(dataset_name)
        prepared = self._get_prepared_dataset(dataset_name, preprocessing_name)
        start = time.perf_counter()
        view = DatasetView(
            dataset_name=dataset_name,
            preprocessing_name=preprocessing_name,
            dataset=dataset,
            split_indices=split_indices,
            train_texts=prepared.processed_texts[split_indices.train_indices],
            val_texts=prepared.processed_texts[split_indices.val_indices],
            test_texts=prepared.processed_texts[split_indices.test_indices],
            train_labels=prepared.labels[split_indices.train_indices],
            val_labels=prepared.labels[split_indices.val_indices],
            test_labels=prepared.labels[split_indices.test_indices],
            test_row_ids=prepared.row_ids[split_indices.test_indices],
        )
        elapsed = time.perf_counter() - start
        self.dataset_view_cache[cache_key] = view
        self._record_profile(
            "build_dataset_view",
            elapsed,
            dataset_name=dataset_name,
            preprocessing_name=preprocessing_name,
            train_rows=len(view.train_labels),
            val_rows=len(view.val_labels),
            test_rows=len(view.test_labels),
        )
        return view

    def _vectorizer_lists(self) -> dict[str, list[Any]]:
        defaults = self._profile_defaults(self.search_profile)
        vectorizer_cfg = self.tuning_config.get("vectorizers", {})
        if vectorizer_cfg and not isinstance(vectorizer_cfg, dict):
            raise ValueError("traditional_ml_tuning.vectorizers must be a mapping when provided.")

        ngram_values = vectorizer_cfg.get("ngram_ranges", defaults["ngram_ranges"])
        ngram_ranges = [tuple(int(item) for item in value) for value in ngram_values]
        max_features = vectorizer_cfg.get("max_features", defaults["max_features"])
        min_df_values = [int(value) for value in vectorizer_cfg.get("min_df", defaults["min_df"])]
        max_df_values = [float(value) for value in vectorizer_cfg.get("max_df", defaults["max_df"])]
        sublinear_tf = [bool(value) for value in vectorizer_cfg.get("sublinear_tf", [False, True])]
        count_binary = [bool(value) for value in vectorizer_cfg.get("count_binary", [False, True])]
        return {
            "ngram_ranges": ngram_ranges,
            "max_features": list(max_features),
            "min_df": min_df_values,
            "max_df": max_df_values,
            "sublinear_tf": sublinear_tf,
            "count_binary": count_binary,
        }

    def _should_keep_vectorizer(self, vectorizer_type: str, config: dict[str, Any]) -> bool:
        ngram_range = tuple(int(value) for value in config["ngram_range"])
        ngram_high = ngram_range[1]
        max_features = config.get("max_features")
        min_df = int(config["min_df"])
        max_df = float(config["max_df"])

        if ngram_high == 3 and max_features is None:
            return False

        if self.search_profile == "quick":
            return ngram_high <= 2

        if self.search_profile == "balanced":
            if ngram_high == 3:
                if max_features not in {10000, 20000, 30000}:
                    return False
                if min_df not in {2, 3, 5}:
                    return False
                if max_df not in {0.95, 0.98}:
                    return False
            if ngram_high == 1 and max_features == 30000:
                return False

        if vectorizer_type == "count" and config.get("binary") is True and ngram_high == 3 and self.search_profile != "full":
            return False

        return True

    def _screen_vectorizer_configs(self) -> list[dict[str, Any]]:
        values = self._vectorizer_lists()
        configs: list[dict[str, Any]] = []
        seen: set[str] = set()

        for vectorizer_type in ("count", "tfidf"):
            for ngram_range in values["ngram_ranges"]:
                for max_features in values["max_features"]:
                    for min_df in values["min_df"]:
                        for max_df in values["max_df"]:
                            base_config = {
                                "type": vectorizer_type,
                                "ngram_range": list(ngram_range),
                                "max_features": max_features,
                                "min_df": int(min_df),
                                "max_df": float(max_df),
                                "lowercase": False,
                            }
                            if vectorizer_type == "count":
                                candidate = {**base_config, "binary": False}
                                if not self._should_keep_vectorizer(vectorizer_type, candidate):
                                    continue
                                signature = _vectorizer_signature(candidate)
                                if signature not in seen:
                                    configs.append(candidate)
                                    seen.add(signature)
                            else:
                                for sublinear_tf in values["sublinear_tf"]:
                                    candidate = {**base_config, "sublinear_tf": bool(sublinear_tf)}
                                    if not self._should_keep_vectorizer(vectorizer_type, candidate):
                                        continue
                                    signature = _vectorizer_signature(candidate)
                                    if signature not in seen:
                                        configs.append(candidate)
                                        seen.add(signature)

        return configs

    def _expand_nb_vectorizers(self, shortlisted_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        values = self._vectorizer_lists()
        expanded: list[dict[str, Any]] = []
        seen: set[str] = set()

        for vectorizer_config in shortlisted_configs:
            if vectorizer_config["type"] == "count":
                for binary_value in values["count_binary"]:
                    candidate = copy.deepcopy(vectorizer_config)
                    candidate["binary"] = bool(binary_value)
                    if not self._should_keep_vectorizer("count", candidate):
                        continue
                    signature = _vectorizer_signature(candidate)
                    if signature not in seen:
                        expanded.append(candidate)
                        seen.add(signature)
            else:
                signature = _vectorizer_signature(vectorizer_config)
                if signature not in seen:
                    expanded.append(copy.deepcopy(vectorizer_config))
                    seen.add(signature)

        return expanded

    def _lr_baseline_config(self) -> dict[str, Any]:
        baseline_cfg = self.tuning_config.get("logistic_regression_baseline", {})
        if baseline_cfg and not isinstance(baseline_cfg, dict):
            raise ValueError("traditional_ml_tuning.logistic_regression_baseline must be a mapping when provided.")

        return {
            "type": "linear",
            "algorithm": "logistic_regression",
            "c": float(baseline_cfg.get("c", 1.0)),
            "penalty": str(baseline_cfg.get("penalty", "l2")).lower(),
            "solver": str(baseline_cfg.get("solver", "liblinear")).lower(),
            "class_weight": baseline_cfg.get("class_weight"),
            "max_iter": int(baseline_cfg.get("max_iter", 5000)),
        }

    def _nb_baseline_config(self) -> dict[str, Any]:
        baseline_cfg = self.tuning_config.get("multinomial_nb_baseline", {})
        if baseline_cfg and not isinstance(baseline_cfg, dict):
            raise ValueError("traditional_ml_tuning.multinomial_nb_baseline must be a mapping when provided.")

        return {
            "type": "multinomial_nb",
            "alpha": float(baseline_cfg.get("alpha", 1.0)),
            "fit_prior": bool(baseline_cfg.get("fit_prior", True)),
        }

    def _logistic_regression_grid(self) -> list[dict[str, Any]]:
        lr_cfg = self.tuning_config.get("logistic_regression", {})
        if lr_cfg and not isinstance(lr_cfg, dict):
            raise ValueError("traditional_ml_tuning.logistic_regression must be a mapping when provided.")

        c_values = [float(value) for value in lr_cfg.get("c", [0.01, 0.1, 1.0, 5.0, 10.0, 20.0])]
        class_weights = list(lr_cfg.get("class_weight", [None, "balanced"]))
        penalties = [str(value).lower() for value in lr_cfg.get("penalty", ["l2", "l1"])]
        solvers = [str(value).lower() for value in lr_cfg.get("solver", ["liblinear", "saga", "lbfgs"])]
        max_iter = int(lr_cfg.get("max_iter", 5000))
        configs: list[dict[str, Any]] = []
        seen: set[str] = set()
        skipped_invalid = 0

        for c_value in c_values:
            for class_weight in class_weights:
                for penalty in penalties:
                    for solver in solvers:
                        if not self._is_valid_logistic_regression_combo(solver=solver, penalty=penalty):
                            skipped_invalid += 1
                            continue
                        config = {
                            "type": "linear",
                            "algorithm": "logistic_regression",
                            "c": float(c_value),
                            "penalty": penalty,
                            "solver": solver,
                            "class_weight": class_weight,
                            "max_iter": max_iter,
                        }
                        signature = _model_signature(config)
                        if signature not in seen:
                            configs.append(config)
                            seen.add(signature)

        if skipped_invalid:
            self.logger.info("Skipped %d invalid Logistic Regression solver/penalty combinations before fitting.", skipped_invalid)
        return configs

    def _naive_bayes_grid(self) -> list[dict[str, Any]]:
        nb_cfg = self.tuning_config.get("multinomial_nb", {})
        if nb_cfg and not isinstance(nb_cfg, dict):
            raise ValueError("traditional_ml_tuning.multinomial_nb must be a mapping when provided.")

        alpha_values = [float(value) for value in nb_cfg.get("alpha", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0])]
        fit_prior_values = [bool(value) for value in nb_cfg.get("fit_prior", [True, False])]
        configs: list[dict[str, Any]] = []
        seen: set[str] = set()

        for alpha_value in alpha_values:
            for fit_prior in fit_prior_values:
                config = {
                    "type": "multinomial_nb",
                    "alpha": float(alpha_value),
                    "fit_prior": bool(fit_prior),
                }
                signature = _model_signature(config)
                if signature not in seen:
                    configs.append(config)
                    seen.add(signature)

        return configs

    @staticmethod
    def _is_valid_logistic_regression_combo(solver: str, penalty: str) -> bool:
        penalty = penalty.lower()
        solver = solver.lower()
        if penalty == "l1":
            return solver in {"liblinear", "saga"}
        if penalty == "l2":
            return solver in {"liblinear", "lbfgs", "saga"}
        return False

    def _base_feature_cache_key(self, view: DatasetView, vectorizer_config: dict[str, Any]) -> tuple[str, str, str]:
        return (view.dataset_name, view.preprocessing_name, _base_vectorizer_signature(vectorizer_config))

    def _feature_cache_key(self, view: DatasetView, vectorizer_config: dict[str, Any]) -> tuple[str, str, str]:
        return (view.dataset_name, view.preprocessing_name, _vectorizer_signature(vectorizer_config))

    def _build_base_feature_bundle(self, view: DatasetView, vectorizer_config: dict[str, Any]) -> BaseFeatureBundle:
        cache_key = self._base_feature_cache_key(view=view, vectorizer_config=vectorizer_config)
        if cache_key in self.base_feature_cache:
            self.cache_stats["base_feature_hit"] += 1
            return self.base_feature_cache[cache_key]

        self.cache_stats["base_feature_miss"] += 1
        base_config = _base_vectorizer_config(vectorizer_config)
        feature_name = slugify(
            f"{view.dataset_name}_{view.preprocessing_name}_base_counts_{_base_vectorizer_signature(vectorizer_config)}"
        )

        start = time.perf_counter()
        vectorizer = CountVectorizer(
            **_base_vectorizer_kwargs(base_config),
            binary=False,
        )
        train_counts = _sorted_csr(vectorizer.fit_transform(view.train_texts))
        val_counts = _sorted_csr(vectorizer.transform(view.val_texts))
        test_counts = _sorted_csr(vectorizer.transform(view.test_texts))
        elapsed = time.perf_counter() - start

        bundle = BaseFeatureBundle(
            feature_name=feature_name,
            base_config=base_config,
            vectorizer=vectorizer,
            train_counts=train_counts,
            val_counts=val_counts,
            test_counts=test_counts,
            build_seconds=float(elapsed),
        )
        self.base_feature_cache[cache_key] = bundle
        self.logger.info(
            "Built base sparse matrices for dataset='%s' preprocessing='%s' n_features=%d in %.2fs",
            view.dataset_name,
            view.preprocessing_name,
            train_counts.shape[1],
            elapsed,
        )
        self._record_profile(
            "build_base_feature_bundle",
            elapsed,
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            feature_dimension=int(train_counts.shape[1]),
            nnz_train=int(train_counts.nnz),
            nnz_val=int(val_counts.nnz),
            nnz_test=int(test_counts.nnz),
            base_signature=_base_vectorizer_signature(vectorizer_config),
        )
        return bundle

    def _get_feature_set(self, view: DatasetView, vectorizer_config: dict[str, Any]) -> FeatureSet:
        cache_key = self._feature_cache_key(view=view, vectorizer_config=vectorizer_config)
        if cache_key in self.feature_cache:
            self.cache_stats["feature_hit"] += 1
            return self.feature_cache[cache_key]

        self.cache_stats["feature_miss"] += 1
        bundle = self._build_base_feature_bundle(view=view, vectorizer_config=vectorizer_config)
        feature_name = slugify(
            f"{view.dataset_name}_{view.preprocessing_name}_{vectorizer_config['type']}_{_vectorizer_signature(vectorizer_config)}"
        )
        variant_start = time.perf_counter()

        if vectorizer_config["type"] == "count":
            binary = bool(vectorizer_config.get("binary", False))
            if binary:
                train_x = _binary_sparse_copy(bundle.train_counts)
                val_x = _binary_sparse_copy(bundle.val_counts)
                test_x = _binary_sparse_copy(bundle.test_counts)
                profile_stage = "derive_binary_count_variant"
            else:
                train_x = bundle.train_counts
                val_x = bundle.val_counts
                test_x = bundle.test_counts
                profile_stage = "reuse_count_variant"

            vectorizer = copy.deepcopy(bundle.vectorizer)
            vectorizer.binary = binary
            artifact = FeatureArtifact(
                name=feature_name,
                feature_type="count",
                transformer=vectorizer,
                metadata={
                    "vocabulary_size": len(bundle.vectorizer.vocabulary_),
                    "config": copy.deepcopy(vectorizer_config),
                    "derived_from_base_signature": _base_vectorizer_signature(vectorizer_config),
                },
            )
        else:
            transformer = TfidfTransformer(
                use_idf=bool(vectorizer_config.get("use_idf", True)),
                smooth_idf=bool(vectorizer_config.get("smooth_idf", True)),
                sublinear_tf=bool(vectorizer_config.get("sublinear_tf", False)),
                norm=vectorizer_config.get("norm", "l2"),
            )
            train_x = _sorted_csr(transformer.fit_transform(bundle.train_counts))
            val_x = _sorted_csr(transformer.transform(bundle.val_counts))
            test_x = _sorted_csr(transformer.transform(bundle.test_counts))
            artifact = FeatureArtifact(
                name=feature_name,
                feature_type="tfidf",
                transformer=Pipeline(
                    [
                        ("vectorizer", copy.deepcopy(bundle.vectorizer)),
                        ("tfidf", transformer),
                    ]
                ),
                metadata={
                    "vocabulary_size": len(bundle.vectorizer.vocabulary_),
                    "config": copy.deepcopy(vectorizer_config),
                    "derived_from_base_signature": _base_vectorizer_signature(vectorizer_config),
                },
            )
            profile_stage = "derive_tfidf_variant"

        feature_set = FeatureSet(train_x=train_x, val_x=val_x, test_x=test_x, artifact=artifact)
        elapsed = time.perf_counter() - variant_start
        self.feature_cache[cache_key] = feature_set
        self._record_profile(
            profile_stage,
            elapsed,
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            vectorizer_type=vectorizer_config["type"],
            vectorizer_signature=_vectorizer_signature(vectorizer_config),
            feature_dimension=int(train_x.shape[1]),
        )
        return feature_set

    def _build_estimator(self, model_name: str, model_config: dict[str, Any], estimator_n_jobs: int | None = None) -> Any:
        if model_name == "logistic_regression":
            penalty = str(model_config.get("penalty", "l2")).lower()
            solver = str(model_config.get("solver", "liblinear")).lower()
            if not self._is_valid_logistic_regression_combo(solver=solver, penalty=penalty):
                raise ValueError(f"Unsupported Logistic Regression combination: penalty={penalty}, solver={solver}")

            return LogisticRegression(
                C=float(model_config["c"]),
                penalty=penalty,
                solver=solver,
                class_weight=model_config.get("class_weight"),
                max_iter=int(model_config.get("max_iter", 5000)),
                random_state=self.seed,
            )

        if model_name == "multinomial_nb":
            return MultinomialNB(
                alpha=float(model_config["alpha"]),
                fit_prior=bool(model_config.get("fit_prior", True)),
            )

        raise ValueError(f"Unsupported model name: {model_name}")

    def _experiment_name(
        self,
        dataset_name: str,
        preprocessing_name: str,
        model_name: str,
        search_stage: str,
        vectorizer_config: dict[str, Any],
        model_config: dict[str, Any],
    ) -> str:
        vectorizer_hash = hashlib.sha1(_vectorizer_signature(vectorizer_config).encode("utf-8")).hexdigest()[:10]
        model_hash = hashlib.sha1(_model_signature(model_config).encode("utf-8")).hexdigest()[:10]
        return slugify(
            f"{dataset_name}_{preprocessing_name}_{model_name}_{search_stage}_{vectorizer_hash}_{model_hash}"
        )

    def _evaluate_trial(
        self,
        view: DatasetView,
        feature_set: FeatureSet,
        vectorizer_config: dict[str, Any],
        model_name: str,
        model_config: dict[str, Any],
        search_stage: str,
        estimator_n_jobs: int | None = None,
    ) -> TrialResult:
        estimator = self._build_estimator(
            model_name=model_name,
            model_config=model_config,
            estimator_n_jobs=estimator_n_jobs,
        )

        start_train = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(feature_set.train_x, view.train_labels)
        train_seconds = time.perf_counter() - start_train

        train_predictions = estimator.predict(feature_set.train_x)
        val_predictions = estimator.predict(feature_set.val_x)

        start_inference = time.perf_counter()
        test_predictions = estimator.predict(feature_set.test_x)
        inference_seconds = time.perf_counter() - start_inference

        train_metrics = compute_classification_metrics(
            view.train_labels,
            train_predictions,
            label_ids=view.dataset.label_ids,
            positive_class_id=view.dataset.positive_class_id,
        )
        val_metrics = compute_classification_metrics(
            view.val_labels,
            val_predictions,
            label_ids=view.dataset.label_ids,
            positive_class_id=view.dataset.positive_class_id,
        )
        test_metrics = compute_classification_metrics(
            view.test_labels,
            test_predictions,
            label_ids=view.dataset.label_ids,
            positive_class_id=view.dataset.positive_class_id,
        )

        warning_messages = [str(item.message) for item in caught_warnings if issubclass(item.category, Warning)]
        selection_score = float(val_metrics[self.selection_metric])
        model_type = "linear" if model_name == "logistic_regression" else "multinomial_nb"

        return TrialResult(
            experiment_name=self._experiment_name(
                dataset_name=view.dataset_name,
                preprocessing_name=view.preprocessing_name,
                model_name=model_name,
                search_stage=search_stage,
                vectorizer_config=vectorizer_config,
                model_config=model_config,
            ),
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            search_stage=search_stage,
            model_name=model_name,
            model_type=model_type,
            vectorizer_config=copy.deepcopy(vectorizer_config),
            model_config=copy.deepcopy(model_config),
            feature_dimension=int(feature_set.train_x.shape[1]),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            train_seconds=float(train_seconds),
            inference_seconds=float(inference_seconds),
            selection_metric=self.selection_metric,
            selection_score=selection_score,
            warning_messages=warning_messages,
        )

    def _evaluate_trial_safe(
        self,
        view: DatasetView,
        feature_set: FeatureSet,
        vectorizer_config: dict[str, Any],
        model_name: str,
        model_config: dict[str, Any],
        search_stage: str,
        estimator_n_jobs: int | None,
    ) -> TrialOutcome:
        try:
            result = self._evaluate_trial(
                view=view,
                feature_set=feature_set,
                vectorizer_config=vectorizer_config,
                model_name=model_name,
                model_config=model_config,
                search_stage=search_stage,
                estimator_n_jobs=estimator_n_jobs,
            )
            return TrialOutcome(result=result, error_message=None, model_config=copy.deepcopy(model_config))
        except Exception as exc:
            return TrialOutcome(result=None, error_message=str(exc), model_config=copy.deepcopy(model_config))

    def _evaluate_model_grid(
        self,
        view: DatasetView,
        feature_set: FeatureSet,
        vectorizer_config: dict[str, Any],
        model_name: str,
        model_grid: list[dict[str, Any]],
        search_stage: str,
    ) -> list[TrialResult]:
        if not model_grid:
            return []

        use_parallel = self.n_jobs != 1 and len(model_grid) >= self.parallel_trial_threshold
        estimator_n_jobs = 1 if use_parallel else self.n_jobs
        start = time.perf_counter()

        if use_parallel:
            outcomes = joblib.Parallel(
                n_jobs=self.n_jobs,
                prefer=self.parallel_backend,
                pre_dispatch=self.pre_dispatch,
            )(
                joblib.delayed(self._evaluate_trial_safe)(
                    view,
                    feature_set,
                    vectorizer_config,
                    model_name,
                    model_config,
                    search_stage,
                    estimator_n_jobs,
                )
                for model_config in model_grid
            )
        else:
            outcomes = [
                self._evaluate_trial_safe(
                    view=view,
                    feature_set=feature_set,
                    vectorizer_config=vectorizer_config,
                    model_name=model_name,
                    model_config=model_config,
                    search_stage=search_stage,
                    estimator_n_jobs=estimator_n_jobs,
                )
                for model_config in model_grid
            ]

        elapsed = time.perf_counter() - start
        self._record_profile(
            "evaluate_model_grid",
            elapsed,
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            model_name=model_name,
            vectorizer_signature=_vectorizer_signature(vectorizer_config),
            trial_count=len(model_grid),
            parallel=use_parallel,
            n_jobs=self.n_jobs if use_parallel else estimator_n_jobs,
        )

        results: list[TrialResult] = []
        for outcome in outcomes:
            if outcome.result is not None:
                results.append(outcome.result)
                continue
            self.logger.warning(
                "Skipping %s tuning trial for dataset='%s' preprocessing='%s': %s | vectorizer=%s | model=%s",
                model_name,
                view.dataset_name,
                view.preprocessing_name,
                outcome.error_message,
                _normalize_for_serialization(vectorizer_config),
                _normalize_for_serialization(outcome.model_config),
            )
        return results

    def _screen_vectorizers(self, view: DatasetView) -> list[TrialResult]:
        screen_configs = self._screen_vectorizer_configs()
        lr_baseline = self._lr_baseline_config()
        nb_baseline = self._nb_baseline_config()
        screen_results: list[TrialResult] = []

        self.logger.info(
            "Vectorizer screening for dataset='%s' preprocessing='%s' with %d vectorizer configuration(s)",
            view.dataset_name,
            view.preprocessing_name,
            len(screen_configs),
        )

        start = time.perf_counter()
        for index, vectorizer_config in enumerate(screen_configs, start=1):
            if index == 1 or index % 25 == 0 or index == len(screen_configs):
                self.logger.info(
                    "Screening vectorizer %d/%d for dataset='%s' preprocessing='%s'",
                    index,
                    len(screen_configs),
                    view.dataset_name,
                    view.preprocessing_name,
                )

            try:
                feature_set = self._get_feature_set(view=view, vectorizer_config=vectorizer_config)
            except Exception as exc:
                self.logger.warning(
                    "Skipping vectorizer config for dataset='%s' preprocessing='%s': %s | config=%s",
                    view.dataset_name,
                    view.preprocessing_name,
                    exc,
                    _normalize_for_serialization(vectorizer_config),
                )
                continue

            for model_name, model_config in (
                ("logistic_regression", lr_baseline),
                ("multinomial_nb", nb_baseline),
            ):
                try:
                    screen_results.append(
                        self._evaluate_trial(
                            view=view,
                            feature_set=feature_set,
                            vectorizer_config=vectorizer_config,
                            model_name=model_name,
                            model_config=model_config,
                            search_stage="vectorizer_screen",
                            estimator_n_jobs=self.n_jobs,
                        )
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Skipping %s screen trial for dataset='%s' preprocessing='%s': %s | vectorizer=%s",
                        model_name,
                        view.dataset_name,
                        view.preprocessing_name,
                        exc,
                        _normalize_for_serialization(vectorizer_config),
                    )

        elapsed = time.perf_counter() - start
        self._record_profile(
            "screen_vectorizers",
            elapsed,
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            screen_vectorizers=len(screen_configs),
            completed_trials=len(screen_results),
        )
        return screen_results

    def _select_top_vectorizers(self, screen_results: list[TrialResult], model_name: str) -> list[dict[str, Any]]:
        shortlisted: list[dict[str, Any]] = []
        seen: set[str] = set()
        model_results = [result for result in screen_results if result.model_name == model_name]

        for result in sorted(model_results, key=_selection_key, reverse=True):
            signature = _vectorizer_signature(result.vectorizer_config)
            if signature in seen:
                continue
            shortlisted.append(copy.deepcopy(result.vectorizer_config))
            seen.add(signature)
            if len(shortlisted) >= self.top_k_vectorizers:
                break

        return shortlisted

    def _tune_model_family(
        self,
        view: DatasetView,
        model_name: str,
        vectorizer_configs: list[dict[str, Any]],
        model_grid: list[dict[str, Any]],
        search_stage: str,
        ) -> list[TrialResult]:
        results: list[TrialResult] = []

        self.logger.info(
            "Hyperparameter search for %s on dataset='%s' preprocessing='%s' with %d vectorizer(s) x %d model config(s)",
            model_name,
            view.dataset_name,
            view.preprocessing_name,
            len(vectorizer_configs),
            len(model_grid),
        )

        completed = 0
        total = len(vectorizer_configs) * len(model_grid)
        family_start = time.perf_counter()

        for vectorizer_config in vectorizer_configs:
            signature = _vectorizer_signature(vectorizer_config)
            try:
                feature_set = self._get_feature_set(view=view, vectorizer_config=vectorizer_config)
            except Exception as exc:
                self.logger.warning(
                    "Skipping cached vectorizer build for %s on dataset='%s' preprocessing='%s': %s | vectorizer=%s",
                    model_name,
                    view.dataset_name,
                    view.preprocessing_name,
                    exc,
                    _normalize_for_serialization(vectorizer_config),
                )
                completed += len(model_grid)
                continue

            batch_results = self._evaluate_model_grid(
                view=view,
                feature_set=feature_set,
                vectorizer_config=vectorizer_config,
                model_name=model_name,
                model_grid=model_grid,
                search_stage=search_stage,
            )
            results.extend(batch_results)
            completed += len(model_grid)

            if completed == len(model_grid) or completed % 50 == 0 or completed == total:
                self.logger.info(
                    "Completed %s trials through vectorizer='%s' progress=%d/%d dataset='%s' preprocessing='%s'",
                    model_name,
                    signature,
                    completed,
                    total,
                    view.dataset_name,
                    view.preprocessing_name,
                )

        elapsed = time.perf_counter() - family_start
        self._record_profile(
            "tune_model_family",
            elapsed,
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            model_name=model_name,
            vectorizer_count=len(vectorizer_configs),
            model_grid_size=len(model_grid),
            completed_results=len(results),
        )
        return results

    def _best_result(self, results: list[TrialResult], model_name: str | None = None) -> TrialResult | None:
        filtered = [result for result in results if model_name is None or result.model_name == model_name]
        if not filtered:
            return None
        return max(filtered, key=_selection_key)

    def _save_predictions(
        self,
        output_slug: str,
        view: DatasetView,
        predictions: np.ndarray,
    ) -> Path | None:
        if not self.output_config.get("save_predictions", False):
            return None

        predictions_path = self.results_run_dir / f"{output_slug}_test_predictions.csv"
        rows = []
        for row_id, true_label, predicted_label in zip(
            view.test_row_ids.tolist(),
            view.test_labels.tolist(),
            predictions.tolist(),
        ):
            rows.append(
                {
                    "row_id": row_id,
                    "y_true": int(true_label),
                    "y_pred": int(predicted_label),
                }
            )
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
        return predictions_path

    def _materialize_best_result(
        self,
        view: DatasetView,
        trial: TrialResult,
        output_label: str,
        leaderboard_csv: Path,
    ) -> dict[str, Any]:
        output_slug = slugify(f"{view.dataset_name}_{view.preprocessing_name}_{output_label}")
        vectorizer_config = copy.deepcopy(trial.vectorizer_config)
        model_config = copy.deepcopy(trial.model_config)
        feature_set = self._get_feature_set(view=view, vectorizer_config=vectorizer_config)
        estimator = self._build_estimator(
            model_name=trial.model_name,
            model_config=model_config,
            estimator_n_jobs=self.n_jobs,
        )

        materialize_start = time.perf_counter()
        start_train = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(feature_set.train_x, view.train_labels)
        train_seconds = time.perf_counter() - start_train

        train_predictions = estimator.predict(feature_set.train_x)
        val_predictions = estimator.predict(feature_set.val_x)
        start_inference = time.perf_counter()
        test_predictions = estimator.predict(feature_set.test_x)
        inference_seconds = time.perf_counter() - start_inference

        train_metrics = compute_classification_metrics(
            view.train_labels,
            train_predictions,
            label_ids=view.dataset.label_ids,
            positive_class_id=view.dataset.positive_class_id,
        )
        val_metrics = compute_classification_metrics(
            view.val_labels,
            val_predictions,
            label_ids=view.dataset.label_ids,
            positive_class_id=view.dataset.positive_class_id,
        )
        test_metrics = compute_classification_metrics(
            view.test_labels,
            test_predictions,
            label_ids=view.dataset.label_ids,
            positive_class_id=view.dataset.positive_class_id,
        )

        feature_artifact_path = self.feature_artifact_dir / f"{output_slug}.joblib"
        model_artifact_path = self.model_artifact_dir / f"{output_slug}.joblib"
        yaml_path = self.experiment_detail_dir / f"{output_slug}.yaml"
        warning_messages = [str(item.message) for item in caught_warnings if issubclass(item.category, Warning)]

        joblib.dump(feature_set.artifact, feature_artifact_path)
        joblib.dump(
            {
                "model": estimator,
                "model_config": model_config,
                "vectorizer_config": vectorizer_config,
                "label_ids": view.dataset.label_ids,
                "positive_class_id": view.dataset.positive_class_id,
            },
            model_artifact_path,
        )

        predictions_path = self._save_predictions(output_slug=output_slug, view=view, predictions=test_predictions)

        detail_payload = {
            "run_id": self.run_id,
            "experiment": {
                "name": output_slug,
                "dataset": view.dataset_name,
                "preprocessing": view.preprocessing_name,
                "feature": f"{vectorizer_config['type']}_traditional_ml_tuned",
                "model": trial.model_name,
            },
            "dataset": {
                "name": view.dataset.name,
                "path": str(view.dataset.path),
                "text_column": view.dataset.text_column,
                "label_column": view.dataset.label_column,
                "id_column": view.dataset.id_column,
                "label_mapping": {str(key): value for key, value in view.dataset.canonical_to_encoded.items()},
                "encoded_to_canonical": {str(key): value for key, value in view.dataset.encoded_to_canonical.items()},
                "positive_class_id": view.dataset.positive_class_id,
            },
            "configs": {
                "dataset_config": _normalize_for_serialization(self.datasets_config[view.dataset_name]),
                "split_config": _normalize_for_serialization(self.datasets_config[view.dataset_name].get("split")),
                "preprocessing_config": _normalize_for_serialization(self.preprocessing_config[view.preprocessing_name]),
                "feature_config": _normalize_for_serialization(vectorizer_config),
                "model_config": _normalize_for_serialization(model_config),
            },
            "split_artifact": str(view.split_indices.artifact_path),
            "feature_artifact": str(feature_artifact_path),
            "model_artifact": str(model_artifact_path),
            "feature_metadata": {
                "feature_type": feature_set.artifact.feature_type,
                "vocabulary_size": feature_set.artifact.metadata.get("vocabulary_size"),
                "embedding_dim": feature_set.artifact.metadata.get("embedding_dim"),
                "max_sequence_length": feature_set.artifact.metadata.get("max_sequence_length"),
                "word2vec_vocabulary_size": feature_set.artifact.metadata.get("word2vec_vocabulary_size"),
                "vocabulary_stats": feature_set.artifact.metadata.get("vocabulary_stats"),
                "artifact_paths": feature_set.artifact.metadata.get("artifact_paths"),
                "diagnostics": feature_set.artifact.metadata.get("diagnostics"),
                "config": _normalize_for_serialization(feature_set.artifact.metadata.get("config", vectorizer_config)),
            },
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
            },
            "timing": {
                "training_time_seconds": float(train_seconds),
                "inference_time_seconds": float(inference_seconds),
            },
            "predictions_path": str(predictions_path) if predictions_path else None,
            "training_history_path": None,
            "extra_metadata": {
                "selection_metric": self.selection_metric,
                "selection_score": float(val_metrics[self.selection_metric]),
                "search_stage": trial.search_stage,
                "leaderboard_csv": str(leaderboard_csv),
                "warning_messages": warning_messages,
            },
        }
        save_yaml(yaml_path, _normalize_for_serialization(detail_payload))

        self._record_profile(
            "materialize_best_result",
            time.perf_counter() - materialize_start,
            dataset_name=view.dataset_name,
            preprocessing_name=view.preprocessing_name,
            model_name=trial.model_name,
            output_label=output_label,
            yaml_path=yaml_path,
        )

        return {
            "model_name": trial.model_name,
            "dataset_name": view.dataset_name,
            "preprocessing_name": view.preprocessing_name,
            "yaml_path": yaml_path,
            "selection_score": float(val_metrics[self.selection_metric]),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "vectorizer_type": vectorizer_config["type"],
            "vectorizer_config": copy.deepcopy(vectorizer_config),
            "model_config": copy.deepcopy(model_config),
        }

    def _save_best_results(
        self,
        view: DatasetView,
        best_lr: TrialResult | None,
        best_nb: TrialResult | None,
        best_overall: TrialResult | None,
        leaderboard_csv: Path,
    ) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        for output_label, trial in (
            ("best_logistic_regression", best_lr),
            ("best_multinomial_nb", best_nb),
            ("best_overall_traditional_ml", best_overall),
        ):
            if trial is None:
                continue
            saved.append(self._materialize_best_result(view=view, trial=trial, output_label=output_label, leaderboard_csv=leaderboard_csv))
        return saved

    def _trial_to_best_record(self, trial: TrialResult) -> dict[str, Any]:
        return {
            "dataset_name": trial.dataset_name,
            "preprocessing_name": trial.preprocessing_name,
            "model_name": trial.model_name,
            "model_type": trial.model_type,
            "search_stage": trial.search_stage,
            "selection_metric": trial.selection_metric,
            "selection_score": float(trial.selection_score),
            "feature_dimension": int(trial.feature_dimension),
            "vectorizer_type": str(trial.vectorizer_config["type"]),
            "vectorizer_signature": _vectorizer_signature(trial.vectorizer_config),
            "vectorizer_config": copy.deepcopy(trial.vectorizer_config),
            "model_config": copy.deepcopy(trial.model_config),
            "train_metrics": copy.deepcopy(trial.train_metrics),
            "validation_metrics": copy.deepcopy(trial.val_metrics),
            "test_metrics": copy.deepcopy(trial.test_metrics),
            "timing": {
                "training_time_seconds": float(trial.train_seconds),
                "inference_time_seconds": float(trial.inference_seconds),
            },
            "warnings": list(trial.warning_messages),
        }

    def _collect_best_by_setup(self, trials: list[TrialResult]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str, str], TrialResult] = {}
        for trial in trials:
            key = (
                trial.dataset_name,
                trial.preprocessing_name,
                trial.model_name,
                _vectorizer_signature(trial.vectorizer_config),
            )
            current = grouped.get(key)
            if current is None or _selection_key(trial) > _selection_key(current):
                grouped[key] = trial

        best_trials = sorted(
            grouped.values(),
            key=lambda trial: (
                trial.dataset_name,
                trial.preprocessing_name,
                trial.model_name,
                trial.vectorizer_config["type"],
                -trial.selection_score,
            ),
        )
        return [self._trial_to_best_record(trial) for trial in best_trials]

    def _resolved_config_payload(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.config)
        tuning_payload = copy.deepcopy(self.tuning_config) if isinstance(self.tuning_config, dict) else {}
        tuning_payload.update(
            {
                "search_profile": self.search_profile,
                "selection_metric": self.selection_metric,
                "top_k_vectorizers": self.top_k_vectorizers,
                "n_jobs": self.n_jobs,
                "parallel_backend": self.parallel_backend,
                "parallel_trial_threshold": self.parallel_trial_threshold,
                "pre_dispatch": self.pre_dispatch,
            }
        )
        payload["traditional_ml_tuning"] = tuning_payload
        return payload

    def run(self) -> dict[str, Any]:
        set_global_seed(self.seed, deterministic_torch=bool(self.runtime_config.get("deterministic_torch", True)))
        selected_datasets = self._selected_names(self.datasets_config, self.dataset_filters, "dataset name(s)")
        selected_preprocessing = self._selected_names(self.preprocessing_config, self.preprocessing_filters, "preprocessing name(s)")

        resolved_config = self._resolved_config_payload()
        resolved_config_path = self.configs_root / f"{self.run_id}_resolved_config.yaml"
        save_yaml(resolved_config_path, _normalize_for_serialization(resolved_config))
        save_yaml(self.results_run_dir / "resolved_config.yaml", _normalize_for_serialization(resolved_config))

        all_trials: list[TrialResult] = []
        saved_best_results: list[dict[str, Any]] = []

        self.logger.info(
            "Starting traditional ML tuning run %s on %d dataset(s) x %d preprocessing profile(s)",
            self.run_id,
            len(selected_datasets),
            len(selected_preprocessing),
        )
        self.logger.info(
            "Parallel trial evaluation configured with n_jobs=%d backend=%s threshold=%d",
            self.n_jobs,
            self.parallel_backend,
            self.parallel_trial_threshold,
        )

        run_start = time.perf_counter()
        for dataset_name in selected_datasets:
            for preprocessing_name in selected_preprocessing:
                group_start = time.perf_counter()
                view = self._build_dataset_view(dataset_name=dataset_name, preprocessing_name=preprocessing_name)
                screen_results = self._screen_vectorizers(view=view)
                if not screen_results:
                    raise RuntimeError(
                        f"No successful vectorizer screening trials for dataset='{dataset_name}' preprocessing='{preprocessing_name}'."
                    )

                lr_vectorizers = self._select_top_vectorizers(screen_results=screen_results, model_name="logistic_regression")
                nb_vectorizers = self._expand_nb_vectorizers(
                    self._select_top_vectorizers(screen_results=screen_results, model_name="multinomial_nb")
                )

                lr_results = self._tune_model_family(
                    view=view,
                    model_name="logistic_regression",
                    vectorizer_configs=lr_vectorizers,
                    model_grid=self._logistic_regression_grid(),
                    search_stage="hyperparameter_search",
                )
                nb_results = self._tune_model_family(
                    view=view,
                    model_name="multinomial_nb",
                    vectorizer_configs=nb_vectorizers,
                    model_grid=self._naive_bayes_grid(),
                    search_stage="hyperparameter_search",
                )

                group_trials = screen_results + lr_results + nb_results
                all_trials.extend(group_trials)

                group_summary_rows = [
                    trial.to_summary_row(
                        run_id=self.run_id,
                        view=view,
                        preprocessing_config=self.preprocessing_config[preprocessing_name],
                    )
                    for trial in group_trials
                ]
                group_frame = pd.DataFrame(group_summary_rows).sort_values(
                    ["selection_score", "val_f1", "val_accuracy", "accuracy", "experiment_name"],
                    ascending=[False, False, False, False, True],
                )
                group_frame["rank_within_dataset_preprocessing"] = np.arange(1, len(group_frame) + 1)
                group_leaderboard_csv = (
                    self.results_run_dir / f"{slugify(dataset_name)}_{slugify(preprocessing_name)}_leaderboard.csv"
                )
                group_frame.to_csv(group_leaderboard_csv, index=False)

                best_lr = self._best_result(group_trials, model_name="logistic_regression")
                best_nb = self._best_result(group_trials, model_name="multinomial_nb")
                best_overall = self._best_result(group_trials)
                saved_best_results.extend(
                    self._save_best_results(
                        view=view,
                        best_lr=best_lr,
                        best_nb=best_nb,
                        best_overall=best_overall,
                        leaderboard_csv=group_leaderboard_csv,
                    )
                )
                self._record_profile(
                    "dataset_preprocessing_group",
                    time.perf_counter() - group_start,
                    dataset_name=dataset_name,
                    preprocessing_name=preprocessing_name,
                    total_trials=len(group_trials),
                )

        summary_rows = []
        for trial in all_trials:
            view = self._build_dataset_view(dataset_name=trial.dataset_name, preprocessing_name=trial.preprocessing_name)
            summary_rows.append(
                trial.to_summary_row(
                    run_id=self.run_id,
                    view=view,
                    preprocessing_config=self.preprocessing_config[trial.preprocessing_name],
                )
            )

        summary_frame = pd.DataFrame(summary_rows)
        if summary_frame.empty:
            raise RuntimeError("No successful trials were produced during tuning.")

        summary_frame = summary_frame.sort_values(
            ["dataset", "preprocessing", "selection_score", "val_f1", "val_accuracy", "accuracy", "experiment_name"],
            ascending=[True, True, False, False, False, False, True],
        ).reset_index(drop=True)
        summary_frame["rank_within_dataset_preprocessing"] = summary_frame.groupby(["dataset", "preprocessing"]).cumcount() + 1
        summary_frame["rank_within_model"] = summary_frame.groupby(["dataset", "preprocessing", "model"]).cumcount() + 1

        summary_csv = self.results_run_dir / "summary.csv"
        summary_json = self.results_run_dir / "summary.json"
        leaderboard_csv = self.results_run_dir / "leaderboard.csv"
        best_results_yaml = self.results_run_dir / "best_results.yaml"
        best_by_setup_yaml = self.results_run_dir / "best_results_by_setup.yaml"
        profiling_csv = self.results_run_dir / "profiling.csv"
        profiling_yaml = self.results_run_dir / "profiling.yaml"
        summary_frame.to_csv(summary_csv, index=False)
        summary_frame.to_csv(leaderboard_csv, index=False)
        save_json(summary_json, {"run_id": self.run_id, "experiments": summary_frame.to_dict(orient="records")})

        best_by_setup_records = self._collect_best_by_setup(all_trials)
        save_yaml(
            best_results_yaml,
            _normalize_for_serialization(
                {
                    "run_id": self.run_id,
                    "selection_metric": self.selection_metric,
                    "materialized_best_results": saved_best_results,
                    "best_by_setup_count": len(best_by_setup_records),
                    "best_by_setup_yaml": str(best_by_setup_yaml),
                }
            ),
        )
        save_yaml(
            best_by_setup_yaml,
            _normalize_for_serialization(
                {
                    "run_id": self.run_id,
                    "selection_metric": self.selection_metric,
                    "records": best_by_setup_records,
                }
            ),
        )

        self._record_profile("traditional_ml_tuning_run", time.perf_counter() - run_start, total_trials=len(all_trials))

        profile_frame = pd.DataFrame(self.profile_records)
        if not profile_frame.empty:
            profile_frame = profile_frame.sort_values("seconds", ascending=False)
        profile_frame.to_csv(profiling_csv, index=False)
        save_yaml(
            profiling_yaml,
            _normalize_for_serialization(
                {
                    "run_id": self.run_id,
                    "cache_stats": dict(self.cache_stats),
                    "events": self.profile_records,
                    "top_bottlenecks": self.profile_records if not self.profile_records else profile_frame.head(25).to_dict(orient="records"),
                }
            ),
        )

        self.logger.info("Completed traditional ML tuning run %s", self.run_id)
        self.logger.info("Summary CSV: %s", summary_csv)
        self.logger.info("Summary JSON: %s", summary_json)
        self.logger.info("Leaderboard CSV: %s", leaderboard_csv)
        self.logger.info("Best results YAML: %s", best_results_yaml)
        self.logger.info("Best-by-setup YAML: %s", best_by_setup_yaml)
        self.logger.info("Profiling CSV: %s", profiling_csv)
        self.logger.info("Profiling YAML: %s", profiling_yaml)
        self.logger.info("Cache stats: %s", dict(self.cache_stats))

        return {
            "run_id": self.run_id,
            "summary_csv": summary_csv,
            "summary_json": summary_json,
            "leaderboard_csv": leaderboard_csv,
            "log_file": self.log_file,
            "resolved_config": resolved_config_path,
            "best_results": saved_best_results,
            "best_results_yaml": best_results_yaml,
            "best_by_setup_yaml": best_by_setup_yaml,
            "profiling_csv": profiling_csv,
            "profiling_yaml": profiling_yaml,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune only traditional ML baselines (Logistic Regression and Multinomial Naive Bayes)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Optional dataset filter. Repeatable.")
    parser.add_argument("--preprocessing", action="append", default=[], help="Optional preprocessing filter. Repeatable.")
    parser.add_argument(
        "--search-profile",
        choices=sorted(VALID_SEARCH_PROFILES),
        default="balanced",
        help="Search size profile. 'quick' is useful for smoke tests, 'full' is the widest search.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=sorted(VALID_SELECTION_METRICS),
        default="f1",
        help="Validation metric used to pick the best configuration.",
    )
    parser.add_argument(
        "--top-k-vectorizers",
        type=int,
        default=None,
        help="Optional override for how many screened vectorizers to keep per model family for stage-two tuning.",
    )
    return parser.parse_args()


def _format_best_result_line(label: str, record: dict[str, Any]) -> str:
    val_metrics = record["val_metrics"]
    test_metrics = record["test_metrics"]
    return (
        f"{label}: dataset={record['dataset_name']} preprocessing={record['preprocessing_name']} "
        f"vectorizer={record['vectorizer_type']} val_f1={val_metrics['f1']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} yaml={record['yaml_path']}"
    )


def main() -> None:
    args = parse_args()
    config, config_path = load_yaml_config(args.config)
    tuner = TraditionalMLTuner(
        config=config,
        config_path=config_path,
        dataset_filters=args.dataset,
        preprocessing_filters=args.preprocessing,
        search_profile=args.search_profile,
        selection_metric=args.selection_metric,
        top_k_vectorizers=args.top_k_vectorizers,
    )
    outputs = tuner.run()

    best_lr = [item for item in outputs["best_results"] if item["yaml_path"].stem.endswith("best_logistic_regression")]
    best_nb = [item for item in outputs["best_results"] if item["yaml_path"].stem.endswith("best_multinomial_nb")]
    best_overall = [item for item in outputs["best_results"] if item["yaml_path"].stem.endswith("best_overall_traditional_ml")]

    print(f"Run completed: {outputs['run_id']}")
    print(f"Summary CSV: {outputs['summary_csv']}")
    print(f"Summary JSON: {outputs['summary_json']}")
    print(f"Leaderboard CSV: {outputs['leaderboard_csv']}")
    print(f"Best results YAML: {outputs['best_results_yaml']}")
    print(f"Best-by-setup YAML: {outputs['best_by_setup_yaml']}")
    print(f"Profiling CSV: {outputs['profiling_csv']}")
    print(f"Profiling YAML: {outputs['profiling_yaml']}")
    print(f"Log file: {outputs['log_file']}")
    print(f"Resolved config: {outputs['resolved_config']}")

    for record in best_lr:
        print(_format_best_result_line("Best Logistic Regression", record))
    for record in best_nb:
        print(_format_best_result_line("Best Naive Bayes", record))
    for record in best_overall:
        print(_format_best_result_line("Best Overall Traditional ML", record))


if __name__ == "__main__":
    main()
