from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.load_batadal import get_batadal_feature_columns, load_batadal_dataset
from data.load_skab import get_skab_feature_columns, load_skab_dataset
from data.preprocessing.noise import add_gaussian_noise
from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from data.preprocessing.sequence import SequenceDataset, generate_sequences
from data.split import split_batadal_by_time, split_features_and_target, split_skab_by_group_holdout


@dataclass
class PreparedSplit:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    sequences: SequenceDataset


@dataclass
class PreparedDataset:
    dataset_name: str
    feature_columns: list[str]
    split_column: str | None
    splits: dict[str, PreparedSplit]


class DataModule:
    """Load, split, preprocess, and window datasets without train/test leakage."""

    def __init__(self, config: dict):
        self.config = config
        self.paths = config["paths"]
        self.preprocessing_config = config["preprocessing"]
        self.sequence_length = self.preprocessing_config.get("sequence_length", 16)
        self.sequence_stride = self.preprocessing_config.get("sequence_stride", 1)

    def prepare_dataset(self, dataset_name: str, scenario: str = "original") -> PreparedDataset:
        dataset_name = dataset_name.lower()
        if dataset_name == "skab":
            return self._prepare_skab(scenario)
        if dataset_name == "batadal":
            return self._prepare_batadal(scenario)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _prepare_skab(self, scenario: str) -> PreparedDataset:
        dataset_config = self.config["datasets"]["skab"]
        dataset = load_skab_dataset(dataset_config, Path(self.paths["raw_data"]))
        feature_columns = get_skab_feature_columns(dataset, dataset_config)
        target_column = dataset_config["target_column"]
        group_column = dataset_config["group_column"]
        raw_splits = split_skab_by_group_holdout(
            dataset=dataset,
            group_column=group_column,
            target_column=target_column,
            split_config=dataset_config["split"],
            random_state=self.config["project"]["random_seeds"][0],
        )
        return self._finalize_dataset(
            dataset_name="skab",
            feature_columns=feature_columns,
            target_column=target_column,
            raw_splits=raw_splits,
            scenario=scenario,
            split_column=group_column,
        )

    def _prepare_batadal(self, scenario: str) -> PreparedDataset:
        dataset_config = self.config["datasets"]["batadal"]
        dataset = load_batadal_dataset(dataset_config, Path(self.paths["raw_data"]))
        feature_columns = get_batadal_feature_columns(dataset, dataset_config)
        target_column = dataset_config["target_column"]
        raw_splits = split_batadal_by_time(dataset, dataset_config["split"])
        return self._finalize_dataset(
            dataset_name="batadal",
            feature_columns=feature_columns,
            target_column=target_column,
            raw_splits=raw_splits,
            scenario=scenario,
            split_column=None,
        )

    def _finalize_dataset(
        self,
        dataset_name: str,
        feature_columns: list[str],
        target_column: str,
        raw_splits: dict[str, pd.DataFrame],
        scenario: str,
        split_column: str | None,
    ) -> PreparedDataset:
        pipeline = PreprocessingPipeline(self.preprocessing_config)

        train_features, train_target = split_features_and_target(raw_splits["train"], feature_columns, target_column)
        train_features = pipeline.fit_transform(train_features).reset_index(drop=True)
        prepared_splits = {
            "train": PreparedSplit(
                frame=raw_splits["train"].copy(),
                features=train_features,
                target=train_target.reset_index(drop=True),
                sequences=generate_sequences(
                    train_features,
                    train_target.reset_index(drop=True),
                    sequence_length=self.sequence_length,
                    stride=self.sequence_stride,
                ),
            )
        }

        for split_name in ("validation", "test"):
            split_features_frame, split_target = split_features_and_target(
                raw_splits[split_name],
                feature_columns,
                target_column,
            )
            transformed_features = pipeline.transform(split_features_frame).reset_index(drop=True)
            if scenario == "noise" and split_name == "test":
                transformed_features = add_gaussian_noise(
                    transformed_features,
                    mean=self.config["noise"].get("gaussian_mean", 0.0),
                    std=self.config["noise"].get("gaussian_std", 0.05),
                    random_state=self.config["project"]["random_seeds"][0],
                )

            prepared_splits[split_name] = PreparedSplit(
                frame=raw_splits[split_name].copy(),
                features=transformed_features,
                target=split_target.reset_index(drop=True),
                sequences=generate_sequences(
                    transformed_features,
                    split_target.reset_index(drop=True),
                    sequence_length=self.sequence_length,
                    stride=self.sequence_stride,
                ),
            )

        return PreparedDataset(
            dataset_name=dataset_name,
            feature_columns=feature_columns,
            split_column=split_column,
            splits=prepared_splits,
        )
