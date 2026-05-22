from __future__ import annotations

import math

import numpy as np


class PAATransformer:
    """Reduce a 1D time series into piecewise means."""

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def transform(self, series: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(series, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("series must contain at least one value")

        segment_count = math.ceil(values.size / self.window_size)
        paa_values = np.empty(segment_count, dtype=float)

        for segment_index in range(segment_count):
            start = segment_index * self.window_size
            end = min(start + self.window_size, values.size)
            paa_values[segment_index] = float(values[start:end].mean())

        return paa_values
