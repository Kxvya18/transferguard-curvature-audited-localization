"""Standardized data structures for temporal graph samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np


@dataclass
class TemporalGraphSample:
    """A single dynamic-graph provenance/localization example."""

    node_ids: list[Hashable]
    adjacency: np.ndarray
    node_features: np.ndarray
    structural_features: np.ndarray
    node_mask: np.ndarray
    origin_index: int
    infection_times: np.ndarray
    observed_mask: np.ndarray
    meta: dict
