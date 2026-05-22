from __future__ import annotations

import numpy as np
import pandas as pd


def add_gaussian_noise(
    features: pd.DataFrame,
    mean: float = 0.0,
    std: float = 0.05,
    random_state: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=mean, scale=std, size=features.shape)
    noisy_values = features.to_numpy(dtype=float) + noise
    return pd.DataFrame(noisy_values, columns=features.columns, index=features.index)
