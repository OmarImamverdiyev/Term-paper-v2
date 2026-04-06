from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.training.data_loading import DatasetBundle
from src.utils.filesystem import save_json


@dataclass
class SplitIndices:
    artifact_name: str
    artifact_path: Path
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def _split_payload(
    dataset: DatasetBundle,
    artifact_name: str,
    seed: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
) -> dict[str, Any]:
    def payload_for(indices: np.ndarray) -> dict[str, Any]:
        return {
            "indices": indices.astype(int).tolist(),
            "row_ids": [dataset.row_ids[index] for index in indices.tolist()],
        }

    return {
        "artifact_name": artifact_name,
        "dataset_name": dataset.name,
        "dataset_path": str(dataset.path),
        "text_column": dataset.text_column,
        "label_column": dataset.label_column,
        "id_column": dataset.id_column,
        "task_type": dataset.task_type,
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_rows": len(dataset.row_ids),
        "splits": {
            "train": payload_for(train_indices),
            "val": payload_for(val_indices),
            "test": payload_for(test_indices),
        },
    }


def _indices_from_row_ids(dataset: DatasetBundle, row_ids: list[str]) -> np.ndarray:
    id_to_index = {row_id: index for index, row_id in enumerate(dataset.row_ids)}
    missing = [row_id for row_id in row_ids if row_id not in id_to_index]
    if missing:
        raise ValueError(
            f"Split reuse failed for dataset '{dataset.name}'. Missing row IDs: {missing[:10]}"
        )
    return np.asarray([id_to_index[row_id] for row_id in row_ids], dtype=np.int64)


def _create_new_split(
    dataset: DatasetBundle,
    split_config: dict[str, Any],
    artifact_path: Path,
) -> SplitIndices:
    train_size = float(split_config.get("train_size", 0.7))
    val_size = float(split_config.get("val_size", 0.15))
    test_size = float(split_config.get("test_size", 0.15))
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    seed = int(split_config.get("seed", 42))
    labels = np.asarray(dataset.labels, dtype=np.int64)
    all_indices = np.arange(len(labels), dtype=np.int64)

    train_indices, temp_indices = train_test_split(
        all_indices,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )

    temp_fraction = val_size + test_size
    val_fraction_within_temp = val_size / temp_fraction
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_fraction_within_temp,
        random_state=seed,
        stratify=labels[temp_indices],
    )

    payload = _split_payload(
        dataset=dataset,
        artifact_name=str(split_config["artifact_name"]),
        seed=seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    save_json(artifact_path, payload)

    return SplitIndices(
        artifact_name=str(split_config["artifact_name"]),
        artifact_path=artifact_path,
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )


def resolve_split_indices(
    dataset: DatasetBundle,
    split_config: dict[str, Any],
    splits_dir: Path,
) -> SplitIndices:
    artifact_name = str(split_config["artifact_name"])
    artifact_path = splits_dir / f"{artifact_name}.json"
    mode = str(split_config.get("mode", "create_or_reuse")).lower()

    if mode not in {"create", "reuse", "create_or_reuse"}:
        raise ValueError(f"Unsupported split mode: {mode}")

    if artifact_path.exists() and mode in {"reuse", "create_or_reuse"}:
        import json

        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        train_indices = _indices_from_row_ids(dataset, payload["splits"]["train"]["row_ids"])
        val_indices = _indices_from_row_ids(dataset, payload["splits"]["val"]["row_ids"])
        test_indices = _indices_from_row_ids(dataset, payload["splits"]["test"]["row_ids"])
        return SplitIndices(
            artifact_name=artifact_name,
            artifact_path=artifact_path,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

    if mode == "reuse":
        raise FileNotFoundError(f"Requested split artifact does not exist: {artifact_path}")

    return _create_new_split(dataset=dataset, split_config=split_config, artifact_path=artifact_path)
