from __future__ import annotations

import string

import numpy as np
from scipy.stats import norm


class SAXTransformer:
    """Map 1D numeric values into a symbolic alphabet using Gaussian breakpoints."""

    def __init__(self, alphabet_size: int):
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2")
        if alphabet_size > len(string.ascii_lowercase):
            raise ValueError("alphabet_size is too large for the available symbol alphabet")

        self.alphabet_size = alphabet_size
        self.alphabet = list(string.ascii_lowercase[:alphabet_size])
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    def transform(self, values: np.ndarray | list[float]) -> list[str]:
        series = np.asarray(values, dtype=float).reshape(-1)
        if series.size == 0:
            raise ValueError("values must contain at least one element")

        mean = float(series.mean())
        std = float(series.std())
        z_normalized = np.zeros_like(series) if np.isclose(std, 0.0) else (series - mean) / std
        indices = np.searchsorted(self.breakpoints, z_normalized, side="right")
        return [self.alphabet[index] for index in indices]
