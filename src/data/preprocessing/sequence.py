from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    features: np.ndarray
    targets: np.ndarray
    sequence_end_indices: np.ndarray


def generate_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int,
    stride: int = 1,
) -> SequenceDataset:
    """Convert row-wise time-series data into overlapping windows."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(features) != len(target):
        raise ValueError("features and target must have the same row count")
    if len(features) < sequence_length:
        raise ValueError("Not enough rows to generate a sequence")

    feature_values = features.to_numpy(dtype=float)
    target_values = target.to_numpy(dtype=int)

    sequences: list[np.ndarray] = []
    targets: list[int] = []
    end_indices: list[int] = []

    for start_index in range(0, len(features) - sequence_length + 1, stride):
        end_index = start_index + sequence_length
        sequences.append(feature_values[start_index:end_index])
        targets.append(int(target_values[start_index:end_index].max()))
        end_indices.append(end_index - 1)

    return SequenceDataset(
        features=np.asarray(sequences, dtype=float),
        targets=np.asarray(targets, dtype=int),
        sequence_end_indices=np.asarray(end_indices, dtype=int),
    )
