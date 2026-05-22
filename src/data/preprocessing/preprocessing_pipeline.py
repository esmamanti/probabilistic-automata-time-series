from __future__ import annotations

import pandas as pd

from data.preprocessing.pca import PCAReducer
from data.preprocessing.scaler import FeatureScaler


class PreprocessingPipeline:
    """Fit preprocessing only on train data and reuse on validation/test."""

    def __init__(self, preprocessing_config: dict):
        self.scaler = FeatureScaler(preprocessing_config.get("scaler", "standard"))
        pca_config = preprocessing_config.get("pca", {})
        self.pca = PCAReducer(
            enabled=pca_config.get("enabled", False),
            n_components=pca_config.get("n_components", 1),
        )

    def fit(self, train_features: pd.DataFrame) -> "PreprocessingPipeline":
        scaled_train = self.scaler.fit_transform(train_features)
        self.pca.fit(scaled_train)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        scaled_train = self.scaler.fit_transform(train_features)
        return self.pca.fit_transform(scaled_train)
