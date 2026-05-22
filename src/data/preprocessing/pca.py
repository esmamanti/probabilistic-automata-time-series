from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, enabled: bool = True, n_components: int = 1):
        self.enabled = enabled
        self.n_components = n_components
        self.reducer = PCA(n_components=n_components) if enabled else None
        self.input_columns_: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "PCAReducer":
        self.input_columns_ = list(features.columns)
        if self.enabled and self.reducer is not None:
            self.reducer.fit(features)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.input_columns_ is None:
            raise RuntimeError("PCA must be fitted before calling transform")
        if not self.enabled or self.reducer is None:
            return features.copy()

        values = self.reducer.transform(features.loc[:, self.input_columns_])
        columns = [f"PC{i + 1}" for i in range(values.shape[1])]
        return pd.DataFrame(values, columns=columns, index=features.index)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
