from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureScaler:
    def __init__(self, scaler_name: str = "standard"):
        scaler_name = scaler_name.lower()
        if scaler_name == "standard":
            self.scaler = StandardScaler()
        elif scaler_name == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {scaler_name}")
        self.feature_names_: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "FeatureScaler":
        self.feature_names_ = list(features.columns)
        self.scaler.fit(features)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform")
        values = self.scaler.transform(features.loc[:, self.feature_names_])
        return pd.DataFrame(values, columns=self.feature_names_, index=features.index)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
