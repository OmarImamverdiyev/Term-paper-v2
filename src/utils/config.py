from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.filesystem import resolve_path


def load_yaml_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved_path = resolve_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {resolved_path}")
    return payload, resolved_path


def get_required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Config key '{key}' must be a non-empty mapping.")
    return value


def get_required_sequence(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config key '{key}' must be a non-empty list.")
    return value


def resolve_optional_path(value: str | Path | None, config_dir: Path) -> Path | None:
    if value is None:
        return None
    return resolve_path(value, config_dir=config_dir)
