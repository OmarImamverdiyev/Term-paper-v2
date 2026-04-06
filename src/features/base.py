from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FeatureArtifact:
    name: str
    feature_type: str
    transformer: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSet:
    train_x: Any
    val_x: Any
    test_x: Any
    artifact: FeatureArtifact
    is_sequence: bool = False
    train_lengths: np.ndarray | None = None
    val_lengths: np.ndarray | None = None
    test_lengths: np.ndarray | None = None
