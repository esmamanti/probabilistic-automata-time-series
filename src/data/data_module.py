from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.load_batadal import get_batadal_feature_columns, load_batadal_dataset
from data.load_skab import get_skab_feature_columns, load_skab_dataset
from data.preprocessing.noise import add_gaussian_noise
from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from data.preprocessing.sequence import SequenceDataset, generate_sequences
from data.split import (
    generate_skab_group_folds,
    split_batadal_by_time,
    split_features_and_target,
    split_skab_by_group_holdout,
    split_skab_train_validation_groups,
)
from utils.seed import get_primary_seed


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
    evaluation_split: str | None = None


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
            random_state=get_primary_seed(self.config),
        )
        return self._finalize_dataset(
            dataset_name="skab",
            feature_columns=feature_columns,
            target_column=target_column,
            raw_splits=raw_splits,
            scenario=scenario,
            split_column=group_column,
        )

    def prepare_skab_fold_datasets(self, scenario: str = "original") -> list[PreparedDataset]:
        dataset_config = self.config["datasets"]["skab"]
        dataset = load_skab_dataset(dataset_config, Path(self.paths["raw_data"]))
        feature_columns = get_skab_feature_columns(dataset, dataset_config)
        target_column = dataset_config["target_column"]
        group_column = dataset_config["group_column"]
        group_count = dataset[group_column].nunique()
        fold_count = min(5, group_count)

        prepared_folds: list[PreparedDataset] = []
        for fold_index, train_idx, test_idx in generate_skab_group_folds(
            dataset=dataset,
            group_column=group_column,
            target_column=target_column,
            n_splits=fold_count,
            random_state=get_primary_seed(self.config),
        ):
            prepared_folds.append(
                self._prepare_skab_fold(
                    dataset=dataset,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    group_column=group_column,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    scenario=scenario,
                    fold_index=fold_index,
                )
            )

        return prepared_folds

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

    def _prepare_skab_fold(
        self,
        dataset: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        group_column: str,
        train_idx,
        test_idx,
        scenario: str,
        fold_index: int,
    ) -> PreparedDataset:
        outer_train = dataset.iloc[train_idx].reset_index(drop=True)
        outer_test = dataset.iloc[test_idx].reset_index(drop=True)
        split_config = self.config["datasets"]["skab"]["split"]
        train_ratio = float(split_config["train"])
        validation_ratio = float(split_config["validation"])
        train_plus_validation = train_ratio + validation_ratio
        if train_plus_validation <= 0:
            raise ValueError("SKAB train and validation ratios must sum to a positive value")
        relative_validation_ratio = validation_ratio / train_plus_validation

        inner_splits = split_skab_train_validation_groups(
            dataset=outer_train,
            group_column=group_column,
            target_column=target_column,
            validation_ratio=relative_validation_ratio,
            random_state=get_primary_seed(self.config) + int(fold_index),
        )
        raw_splits = {
            "train": inner_splits["train"],
            "validation": inner_splits["validation"],
            "test": outer_test,
        }
        return self._finalize_dataset(
            dataset_name="skab",
            feature_columns=feature_columns,
            target_column=target_column,
            raw_splits=raw_splits,
            scenario=scenario,
            split_column=group_column,
            evaluation_split=f"fold_{fold_index}",
        )

    def _finalize_dataset(
        self,
        dataset_name: str,
        feature_columns: list[str],
        target_column: str,
        raw_splits: dict[str, pd.DataFrame],
        scenario: str,
        split_column: str | None,
        evaluation_split: str | None = None,
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
                    random_state=get_primary_seed(self.config),
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
            evaluation_split=evaluation_split,
        )
