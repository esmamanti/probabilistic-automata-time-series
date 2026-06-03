from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from data.preprocessing.pca import PCAReducer
from data.preprocessing.scaler import FeatureScaler


class PreprocessingPipeline:
    """Fit preprocessing only on train data and reuse on validation/test."""

    def __init__(self, preprocessing_config: dict):
        missing_data_config = preprocessing_config.get("missing_data", {})
        self.missing_data_enabled = bool(missing_data_config.get("enabled", True))
        self.imputer = SimpleImputer(strategy=missing_data_config.get("strategy", "mean"))
        self.scaler = FeatureScaler(preprocessing_config.get("scaler", "standard"))
        pca_config = preprocessing_config.get("pca", {})
        self.pca = PCAReducer(
            enabled=pca_config.get("enabled", False),
            n_components=pca_config.get("n_components", 1),
        )
        self.feature_columns: list[str] = []

    @staticmethod
    def summarize_missing_values(features: pd.DataFrame) -> dict[str, object]:
        missing_per_column = features.isna().sum()
        columns_with_missing = missing_per_column[missing_per_column > 0]
        return {
            "row_count": int(len(features)),
            "column_count": int(len(features.columns)),
            "missing_value_count": int(missing_per_column.sum()),
            "columns_with_missing": {column: int(count) for column, count in columns_with_missing.items()},
        }

    def _impute(self, features: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        self.feature_columns = list(features.columns)
        if not self.missing_data_enabled:
            return features.copy()

        if fit:
            transformed = self.imputer.fit_transform(features)
        else:
            transformed = self.imputer.transform(features)
        return pd.DataFrame(transformed, columns=self.feature_columns, index=features.index)

    def fit(self, train_features: pd.DataFrame) -> "PreprocessingPipeline":
        imputed_train = self._impute(train_features, fit=True)
        scaled_train = self.scaler.fit_transform(imputed_train)
        self.pca.fit(scaled_train)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        imputed = self._impute(features, fit=False)
        scaled = self.scaler.transform(imputed)
        return self.pca.transform(scaled)

    def fit_transform(self, train_features: pd.DataFrame) -> pd.DataFrame:
        imputed_train = self._impute(train_features, fit=True)
        scaled_train = self.scaler.fit_transform(imputed_train)
        return self.pca.fit_transform(scaled_train)
