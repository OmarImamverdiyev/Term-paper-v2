from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.filesystem import resolve_path


def _normalize_label_key(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _build_raw_label_mapper(label_map: dict[Any, Any] | None) -> dict[Any, Any]:
    if not label_map:
        return {}
    mapper: dict[Any, Any] = {}
    for raw_key, mapped_value in label_map.items():
        normalized_key = _normalize_label_key(raw_key)
        mapper[normalized_key] = mapped_value
        mapper[str(normalized_key)] = mapped_value
    return mapper


@dataclass
class DatasetBundle:
    name: str
    path: Path
    frame: pd.DataFrame
    texts: list[str]
    labels: list[int]
    row_ids: list[str]
    label_ids: list[int]
    encoded_to_canonical: dict[int, Any]
    canonical_to_encoded: dict[Any, int]
    text_column: str
    label_column: str
    id_column: str | None
    positive_class_id: int | None
    task_type: str


def load_csv_dataset(
    dataset_name: str,
    dataset_config: dict[str, Any],
    config_dir: Path,
    seed: int,
) -> DatasetBundle:
    dataset_path = resolve_path(dataset_config["path"], config_dir=config_dir)
    frame = pd.read_csv(dataset_path)

    text_column = str(dataset_config["text_column"])
    label_column = str(dataset_config["label_column"])
    id_column = dataset_config.get("id_column")
    task_type = str(dataset_config.get("task_type", "binary")).lower()

    missing_columns = [column for column in [text_column, label_column, id_column] if column and column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset '{dataset_name}' is missing columns: {missing_columns}")

    frame = frame.copy()
    frame[text_column] = frame[text_column].fillna("").astype(str)
    frame = frame[frame[text_column].str.strip().astype(bool)].copy()
    frame = frame.dropna(subset=[label_column]).copy()
    frame.reset_index(drop=True, inplace=True)

    if id_column:
        frame["__row_id__"] = frame[id_column].astype(str)
    else:
        frame["__row_id__"] = [f"row_{index}" for index in range(len(frame))]

    raw_mapper = _build_raw_label_mapper(dataset_config.get("label_map"))
    raw_labels = frame[label_column].map(_normalize_label_key)
    if raw_mapper:
        canonical_labels = raw_labels.map(lambda value: raw_mapper.get(value, raw_mapper.get(str(value))))
        if canonical_labels.isna().any():
            unmapped = raw_labels[canonical_labels.isna()].unique().tolist()
            raise ValueError(
                f"Dataset '{dataset_name}' has unmapped labels in '{label_column}': {unmapped}"
            )
    else:
        canonical_labels = raw_labels

    canonical_values = sorted(canonical_labels.dropna().unique().tolist(), key=lambda value: str(value))
    if task_type == "binary" and len(canonical_values) != 2:
        raise ValueError(
            f"Dataset '{dataset_name}' is configured as binary but has {len(canonical_values)} classes."
        )
    if len(canonical_values) < 2:
        raise ValueError(f"Dataset '{dataset_name}' must contain at least two classes.")

    canonical_to_encoded = {value: index for index, value in enumerate(canonical_values)}
    encoded_to_canonical = {index: value for value, index in canonical_to_encoded.items()}
    frame["__label__"] = canonical_labels.map(canonical_to_encoded).astype(int)

    sample_size = dataset_config.get("sample_size")
    if sample_size:
        sample_size = int(sample_size)
        if len(frame) > sample_size:
            if sample_size < len(canonical_values):
                raise ValueError(
                    f"Sample size {sample_size} for dataset '{dataset_name}' is smaller than class count."
                )
            frame, _unused = train_test_split(
                frame,
                train_size=sample_size,
                random_state=seed,
                stratify=frame["__label__"],
            )
            frame = frame.sort_index().reset_index(drop=True)

    positive_class_id: int | None = None
    if len(canonical_values) == 2:
        configured_positive = dataset_config.get("positive_label")
        if configured_positive is None:
            positive_class_id = max(canonical_to_encoded.values())
        else:
            normalized_positive = _normalize_label_key(configured_positive)
            if normalized_positive not in canonical_to_encoded:
                raise ValueError(
                    f"Positive label '{configured_positive}' is not present in dataset '{dataset_name}'."
                )
            positive_class_id = canonical_to_encoded[normalized_positive]

    return DatasetBundle(
        name=dataset_name,
        path=dataset_path,
        frame=frame.reset_index(drop=True),
        texts=frame[text_column].astype(str).tolist(),
        labels=frame["__label__"].astype(int).tolist(),
        row_ids=frame["__row_id__"].astype(str).tolist(),
        label_ids=list(range(len(canonical_values))),
        encoded_to_canonical=encoded_to_canonical,
        canonical_to_encoded=canonical_to_encoded,
        text_column=text_column,
        label_column=label_column,
        id_column=str(id_column) if id_column else None,
        positive_class_id=positive_class_id,
        task_type=task_type,
    )
