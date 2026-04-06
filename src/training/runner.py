from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.base import FeatureSet
from src.features.pmi import build_pmi_features
from src.features.vectorizers import build_count_features, build_tfidf_features
from src.features.word2vec import build_word2vec_features
from src.preprocessing.text_cleaner import TextPreprocessor
from src.training.data_loading import DatasetBundle, load_csv_dataset
from src.training.splits import SplitIndices, resolve_split_indices
from src.training.trainers import (
    ExperimentOutcome,
    train_sklearn_experiment,
    train_torch_experiment,
    validate_experiment_compatibility,
)
from src.utils.config import get_required_mapping, get_required_sequence
from src.utils.filesystem import ensure_directories, make_run_id, resolve_path, save_json, save_yaml, slugify
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_global_seed


@dataclass
class PreparedDataset:
    dataset: DatasetBundle
    processed_texts: list[str]
    token_lists: list[list[str]]


class ExperimentRunner:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        experiment_filters: dict[str, set[str]] | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.config_dir = config_path.parent
        self.filters = experiment_filters or {}

        self.datasets_config = get_required_mapping(config, "datasets")
        self.preprocessing_config = get_required_mapping(config, "preprocessing")
        self.features_config = get_required_mapping(config, "features")
        self.models_config = get_required_mapping(config, "models")
        self.experiments_config = get_required_sequence(config, "experiments")

        self.seed = int(config.get("seed", 42))
        self.runtime_config = config.get("runtime", {})
        self.output_config = config.get("output", {})
        self.device = self._resolve_device()

        self.datasets_cache: dict[str, DatasetBundle] = {}
        self.splits_cache: dict[str, SplitIndices] = {}
        self.preprocessed_cache: dict[tuple[str, str], PreparedDataset] = {}
        self.features_cache: dict[tuple[str, str, str], tuple[FeatureSet, Path]] = {}

        self.run_id = make_run_id(str(self.output_config.get("run_name", self.config_path.stem)))
        self.results_root = resolve_path(self.output_config.get("results_dir", "results"), config_dir=self.config_dir)
        self.logs_root = resolve_path(self.output_config.get("logs_dir", "logs"), config_dir=self.config_dir)
        self.models_root = resolve_path(self.output_config.get("models_dir", "models"), config_dir=self.config_dir)
        self.splits_root = resolve_path(self.output_config.get("splits_dir", "splits"), config_dir=self.config_dir)
        self.configs_root = resolve_path(self.output_config.get("configs_dir", "configs"), config_dir=self.config_dir)
        self.word2vec_root = resolve_path(self.output_config.get("word2vec_dir", "word2vec"), config_dir=self.config_dir)

        ensure_directories(
            [self.results_root, self.logs_root, self.models_root, self.splits_root, self.configs_root, self.word2vec_root]
        )

        self.results_run_dir = self.results_root / self.run_id
        self.model_run_dir = self.models_root / self.run_id
        self.word2vec_run_dir = self.word2vec_root / self.run_id
        self.feature_artifact_dir = self.model_run_dir / "feature_artifacts"
        self.model_artifact_dir = self.model_run_dir / "model_artifacts"
        self.experiment_detail_dir = self.results_run_dir / "experiment_details"
        self.training_curve_dir = self.results_run_dir / "training_curves"
        ensure_directories(
            [
                self.results_run_dir,
                self.word2vec_run_dir,
                self.feature_artifact_dir,
                self.model_artifact_dir,
                self.experiment_detail_dir,
                self.training_curve_dir,
            ]
        )

        self.log_file = self.logs_root / f"{self.run_id}.log"
        self.logger = setup_logger(self.log_file)

    def _resolve_device(self) -> torch.device:
        configured = str(self.runtime_config.get("device", "auto")).lower()
        if configured == "cpu":
            return torch.device("cpu")
        if configured == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested in config but not available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _selected_experiments(self) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        dataset_filters = self.filters.get("datasets", set())
        feature_filters = self.filters.get("features", set())
        model_filters = self.filters.get("models", set())
        experiment_name_filters = self.filters.get("experiments", set())

        for experiment in self.experiments_config:
            if not isinstance(experiment, dict):
                raise ValueError("Every experiment entry must be a mapping.")
            name = str(experiment.get("name") or f"{experiment['dataset']}_{experiment['feature']}_{experiment['model']}")
            if dataset_filters and str(experiment["dataset"]) not in dataset_filters:
                continue
            if feature_filters and str(experiment["feature"]) not in feature_filters:
                continue
            if model_filters and str(experiment["model"]) not in model_filters:
                continue
            if experiment_name_filters and name not in experiment_name_filters:
                continue
            item = copy.deepcopy(experiment)
            item["name"] = name
            selected.append(item)

        if not selected:
            raise ValueError("No experiments matched the current config and CLI filters.")
        return selected

    def _get_dataset(self, dataset_name: str) -> DatasetBundle:
        if dataset_name not in self.datasets_cache:
            dataset_config = self.datasets_config[dataset_name]
            dataset = load_csv_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                config_dir=self.config_dir,
                seed=self.seed,
            )
            self.datasets_cache[dataset_name] = dataset
            self.logger.info("Loaded dataset '%s' with %d rows from %s", dataset_name, len(dataset.texts), dataset.path)
        return self.datasets_cache[dataset_name]

    def _get_split_indices(self, dataset_name: str) -> SplitIndices:
        if dataset_name not in self.splits_cache:
            dataset = self._get_dataset(dataset_name)
            dataset_config = self.datasets_config[dataset_name]
            split_config = dataset_config.get("split")
            if not isinstance(split_config, dict) or "artifact_name" not in split_config:
                raise ValueError(f"Dataset '{dataset_name}' must define a split config with 'artifact_name'.")
            split_indices = resolve_split_indices(dataset=dataset, split_config=split_config, splits_dir=self.splits_root)
            self.splits_cache[dataset_name] = split_indices
            self.logger.info(
                "Resolved split '%s' for dataset '%s' -> train=%d val=%d test=%d",
                split_indices.artifact_name,
                dataset_name,
                len(split_indices.train_indices),
                len(split_indices.val_indices),
                len(split_indices.test_indices),
            )
            self._validate_split_indices(dataset=dataset, split_indices=split_indices)
        return self.splits_cache[dataset_name]

    @staticmethod
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

    def _get_prepared_dataset(self, dataset_name: str, preprocessing_name: str) -> PreparedDataset:
        cache_key = (dataset_name, preprocessing_name)
        if cache_key not in self.preprocessed_cache:
            dataset = self._get_dataset(dataset_name)
            preprocessing_cfg = self.preprocessing_config[preprocessing_name]
            preprocessor = TextPreprocessor(config=preprocessing_cfg, config_dir=self.config_dir)
            processed_texts, token_lists = preprocessor.preprocess_many(dataset.texts)
            prepared = PreparedDataset(dataset=dataset, processed_texts=processed_texts, token_lists=token_lists)
            self.preprocessed_cache[cache_key] = prepared
            self.logger.info(
                "Prepared dataset '%s' using preprocessing '%s'",
                dataset_name,
                preprocessing_name,
            )
        return self.preprocessed_cache[cache_key]

    @staticmethod
    def _slice_list(items: list[Any], indices: np.ndarray) -> list[Any]:
        return [items[int(index)] for index in indices.tolist()]

    def _build_feature_set(
        self,
        dataset_name: str,
        preprocessing_name: str,
        feature_name: str,
    ) -> tuple[FeatureSet, Path]:
        cache_key = (dataset_name, preprocessing_name, feature_name)
        if cache_key in self.features_cache:
            return self.features_cache[cache_key]

        feature_cfg = self.features_config[feature_name]
        feature_type = str(feature_cfg.get("type", "")).lower()
        prepared = self._get_prepared_dataset(dataset_name, preprocessing_name)
        split_indices = self._get_split_indices(dataset_name)

        train_texts = self._slice_list(prepared.processed_texts, split_indices.train_indices)
        val_texts = self._slice_list(prepared.processed_texts, split_indices.val_indices)
        test_texts = self._slice_list(prepared.processed_texts, split_indices.test_indices)

        train_tokens = self._slice_list(prepared.token_lists, split_indices.train_indices)
        val_tokens = self._slice_list(prepared.token_lists, split_indices.val_indices)
        test_tokens = self._slice_list(prepared.token_lists, split_indices.test_indices)

        train_labels = np.asarray(self._slice_list(prepared.dataset.labels, split_indices.train_indices), dtype=np.int64)
        feature_slug = slugify(f"{dataset_name}_{preprocessing_name}_{feature_name}")

        if feature_type == "count":
            feature_set = build_count_features(feature_name, feature_cfg, train_texts, val_texts, test_texts)
        elif feature_type == "tfidf":
            feature_set = build_tfidf_features(feature_name, feature_cfg, train_texts, val_texts, test_texts)
        elif feature_type == "pmi":
            feature_set = build_pmi_features(
                feature_name,
                feature_cfg,
                train_texts,
                val_texts,
                test_texts,
                train_labels=train_labels,
            )
        elif feature_type == "word2vec":
            feature_set = build_word2vec_features(
                feature_name,
                feature_cfg,
                train_tokens,
                val_tokens,
                test_tokens,
                seed=self.seed,
                artifact_dir=self.word2vec_run_dir,
                artifact_prefix=feature_slug,
            )
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        feature_artifact_path = self.feature_artifact_dir / f"{feature_slug}.joblib"
        joblib.dump(feature_set.artifact, feature_artifact_path)

        self.features_cache[cache_key] = (feature_set, feature_artifact_path)
        self.logger.info(
            "Built feature '%s' for dataset '%s' using preprocessing '%s'",
            feature_name,
            dataset_name,
            preprocessing_name,
        )
        if feature_type == "word2vec":
            metadata = feature_set.artifact.metadata
            vocabulary_stats = metadata.get("vocabulary_stats", {})
            self.logger.info(
                "Word2Vec trained on train-only corpus: docs=%s tokens=%s w2v_vocab=%s seq_vocab=%s max_seq_len=%s",
                vocabulary_stats.get("train_corpus", {}).get("documents"),
                vocabulary_stats.get("train_corpus", {}).get("tokens"),
                metadata.get("word2vec_vocabulary_size"),
                metadata.get("vocabulary_size"),
                metadata.get("max_sequence_length"),
            )
            for diagnostic in metadata.get("diagnostics", []):
                self.logger.warning("Word2Vec diagnostic: %s", diagnostic)
        return feature_set, feature_artifact_path

    def _summary_row(
        self,
        experiment: dict[str, Any],
        dataset: DatasetBundle,
        split_indices: SplitIndices,
        feature_set: FeatureSet,
        feature_artifact_path: Path,
        outcome: ExperimentOutcome,
        model_path: Path,
        metadata_path: Path,
    ) -> dict[str, Any]:
        test_metrics = outcome.test_metrics
        val_metrics = outcome.val_metrics
        feature_metadata = feature_set.artifact.metadata
        return {
            "run_id": self.run_id,
            "experiment_name": experiment["name"],
            "dataset": experiment["dataset"],
            "preprocessing": experiment["preprocessing"],
            "feature": experiment["feature"],
            "feature_type": feature_set.artifact.feature_type,
            "model": experiment["model"],
            "model_type": self.models_config[experiment["model"]]["type"],
            "dataset_path": str(dataset.path),
            "split_artifact": str(split_indices.artifact_path),
            "feature_artifact": str(feature_artifact_path),
            "model_artifact": str(model_path),
            "metadata_path": str(metadata_path),
            "train_rows": len(split_indices.train_indices),
            "val_rows": len(split_indices.val_indices),
            "test_rows": len(split_indices.test_indices),
            "feature_dimension": int(
                feature_metadata.get("embedding_dim", feature_set.train_x.shape[1]) if feature_set.is_sequence else feature_set.train_x.shape[1]
            ),
            "sequence_vocabulary_size": feature_metadata.get("vocabulary_size") if feature_set.is_sequence else None,
            "max_sequence_length": feature_metadata.get("max_sequence_length") if feature_set.is_sequence else None,
            "word2vec_artifacts_json": json.dumps(feature_metadata.get("artifact_paths", {})) if feature_set.is_sequence else None,
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "confusion_matrix_json": json.dumps(test_metrics["confusion_matrix"]),
            "training_time_seconds": outcome.train_seconds,
            "inference_time_seconds": outcome.inference_seconds,
        }

    def _save_predictions(
        self,
        experiment_slug: str,
        dataset: DatasetBundle,
        split_indices: SplitIndices,
        outcome: ExperimentOutcome,
    ) -> Path | None:
        if not self.output_config.get("save_predictions", False):
            return None

        predictions_path = self.results_run_dir / f"{experiment_slug}_test_predictions.csv"
        rows = []
        test_row_ids = self._slice_list(dataset.row_ids, split_indices.test_indices)
        true_labels = self._slice_list(dataset.labels, split_indices.test_indices)

        model_predictions = outcome.extra_metadata.get("test_predictions")
        if model_predictions is None:
            return None

        for row_id, true_label, predicted_label in zip(test_row_ids, true_labels, model_predictions):
            rows.append(
                {
                    "row_id": row_id,
                    "y_true": int(true_label),
                    "y_pred": int(predicted_label),
                }
            )
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
        return predictions_path

    def _save_training_history(self, experiment_slug: str, outcome: ExperimentOutcome) -> Path | None:
        history = outcome.extra_metadata.get("history")
        if not history:
            return None

        history_path = self.training_curve_dir / f"{experiment_slug}.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        return history_path

    def run(self) -> dict[str, Path]:
        set_global_seed(self.seed, deterministic_torch=bool(self.runtime_config.get("deterministic_torch", True)))
        selected_experiments = self._selected_experiments()
        self.logger.info("Starting run %s with %d experiment(s) on device=%s", self.run_id, len(selected_experiments), self.device)

        resolved_config_path = self.configs_root / f"{self.run_id}_resolved_config.yaml"
        save_yaml(resolved_config_path, self.config)
        save_yaml(self.results_run_dir / "resolved_config.yaml", self.config)

        summary_rows: list[dict[str, Any]] = []

        for experiment in selected_experiments:
            dataset_name = str(experiment["dataset"])
            preprocessing_name = str(experiment["preprocessing"])
            feature_name = str(experiment["feature"])
            model_name = str(experiment["model"])

            dataset = self._get_dataset(dataset_name)
            split_indices = self._get_split_indices(dataset_name)
            feature_set, feature_artifact_path = self._build_feature_set(dataset_name, preprocessing_name, feature_name)

            model_config = self.models_config[model_name]
            model_type = str(model_config.get("type", "")).lower()
            validate_experiment_compatibility(feature_set.artifact.feature_type, model_type)

            train_labels = np.asarray(self._slice_list(dataset.labels, split_indices.train_indices), dtype=np.int64)
            val_labels = np.asarray(self._slice_list(dataset.labels, split_indices.val_indices), dtype=np.int64)
            test_labels = np.asarray(self._slice_list(dataset.labels, split_indices.test_indices), dtype=np.int64)

            experiment_slug = slugify(experiment["name"])
            model_extension = ".pt" if model_type in {"mlp", "birnn", "lstm"} else ".joblib"
            model_path = self.model_artifact_dir / f"{experiment_slug}{model_extension}"

            self.logger.info(
                "Running experiment '%s' [%s + %s + %s]",
                experiment["name"],
                dataset_name,
                feature_name,
                model_name,
            )

            if model_type in {"multinomial_nb", "linear"}:
                outcome = train_sklearn_experiment(
                    model_config=model_config,
                    feature_set=feature_set,
                    train_labels=train_labels,
                    val_labels=val_labels,
                    test_labels=test_labels,
                    label_ids=dataset.label_ids,
                    positive_class_id=dataset.positive_class_id,
                    seed=self.seed,
                    model_output_path=model_path,
                )
            else:
                outcome = train_torch_experiment(
                    model_config=model_config,
                    feature_set=feature_set,
                    train_labels=train_labels,
                    val_labels=val_labels,
                    test_labels=test_labels,
                    label_ids=dataset.label_ids,
                    positive_class_id=dataset.positive_class_id,
                    seed=self.seed,
                    device=self.device,
                    model_output_path=model_path,
                    logger=self.logger,
                )

            predictions_path = self._save_predictions(
                experiment_slug=experiment_slug,
                dataset=dataset,
                split_indices=split_indices,
                outcome=outcome,
            )
            training_history_path = self._save_training_history(experiment_slug=experiment_slug, outcome=outcome)

            metadata_path = self.experiment_detail_dir / f"{experiment_slug}.json"
            extra_metadata = {key: value for key, value in outcome.extra_metadata.items() if key != "test_predictions"}
            metadata = {
                "run_id": self.run_id,
                "experiment": experiment,
                "dataset": {
                    "name": dataset.name,
                    "path": str(dataset.path),
                    "text_column": dataset.text_column,
                    "label_column": dataset.label_column,
                    "id_column": dataset.id_column,
                    "label_mapping": {str(key): value for key, value in dataset.canonical_to_encoded.items()},
                    "encoded_to_canonical": {str(key): value for key, value in dataset.encoded_to_canonical.items()},
                    "positive_class_id": dataset.positive_class_id,
                },
                "configs": {
                    "dataset_config": self.datasets_config[dataset_name],
                    "split_config": self.datasets_config[dataset_name].get("split"),
                    "preprocessing_config": self.preprocessing_config[preprocessing_name],
                    "feature_config": self.features_config[feature_name],
                    "model_config": model_config,
                },
                "split_artifact": str(split_indices.artifact_path),
                "feature_artifact": str(feature_artifact_path),
                "model_artifact": str(model_path),
                "feature_metadata": {
                    "feature_type": feature_set.artifact.feature_type,
                    "vocabulary_size": feature_set.artifact.metadata.get("vocabulary_size"),
                    "embedding_dim": feature_set.artifact.metadata.get("embedding_dim"),
                    "max_sequence_length": feature_set.artifact.metadata.get("max_sequence_length"),
                    "word2vec_vocabulary_size": feature_set.artifact.metadata.get("word2vec_vocabulary_size"),
                    "vocabulary_stats": feature_set.artifact.metadata.get("vocabulary_stats"),
                    "artifact_paths": feature_set.artifact.metadata.get("artifact_paths"),
                    "diagnostics": feature_set.artifact.metadata.get("diagnostics"),
                },
                "metrics": {
                    "train": outcome.train_metrics,
                    "validation": outcome.val_metrics,
                    "test": outcome.test_metrics,
                },
                "timing": {
                    "training_time_seconds": outcome.train_seconds,
                    "inference_time_seconds": outcome.inference_seconds,
                },
                "predictions_path": str(predictions_path) if predictions_path else None,
                "training_history_path": str(training_history_path) if training_history_path else None,
                "extra_metadata": extra_metadata,
            }
            save_json(metadata_path, metadata)

            summary_rows.append(
                self._summary_row(
                    experiment=experiment,
                    dataset=dataset,
                    split_indices=split_indices,
                    feature_set=feature_set,
                    feature_artifact_path=feature_artifact_path,
                    outcome=outcome,
                    model_path=model_path,
                    metadata_path=metadata_path,
                )
            )

            self.logger.info(
                "Finished experiment '%s' with test macro_f1=%.4f accuracy=%.4f",
                experiment["name"],
                outcome.test_metrics["macro_f1"],
                outcome.test_metrics["accuracy"],
            )

        summary_frame = pd.DataFrame(summary_rows)
        summary_csv = self.results_run_dir / "summary.csv"
        summary_json = self.results_run_dir / "summary.json"
        summary_frame.to_csv(summary_csv, index=False)
        save_json(summary_json, {"run_id": self.run_id, "experiments": summary_rows})

        self.logger.info("Completed run %s", self.run_id)
        self.logger.info("Summary CSV: %s", summary_csv)
        self.logger.info("Summary JSON: %s", summary_json)

        return {
            "summary_csv": summary_csv,
            "summary_json": summary_json,
            "log_file": self.log_file,
            "resolved_config": resolved_config_path,
        }
