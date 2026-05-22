from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def generate_skab_group_folds(
    dataset: pd.DataFrame,
    group_column: str,
    target_column: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield fold index together with train/test indices for SKAB."""
    if group_column not in dataset.columns:
        raise KeyError(f"Group column '{group_column}' not found")
    if target_column not in dataset.columns:
        raise KeyError(f"Target column '{target_column}' not found")

    splitter = None
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        iterator = splitter.split(dataset, dataset[target_column], dataset[group_column])
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(dataset, groups=dataset[group_column])

    for fold_index, (train_idx, test_idx) in enumerate(iterator):
        yield fold_index, train_idx, test_idx


def split_batadal_by_time(dataset: pd.DataFrame, split_config: dict) -> dict[str, pd.DataFrame]:
    """Split BATADAL into contiguous train/validation/test partitions."""
    train_ratio = split_config["train"]
    validation_ratio = split_config["validation"]
    test_ratio = split_config["test"]

    total_ratio = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"BATADAL split ratios must sum to 1.0, got {total_ratio}")

    total_rows = len(dataset)
    if total_rows < 3:
        raise ValueError("BATADAL dataset must contain at least 3 rows for train/validation/test split")

    train_end = int(total_rows * train_ratio)
    validation_end = train_end + int(total_rows * validation_ratio)

    if train_end <= 0 or validation_end <= train_end or validation_end >= total_rows:
        raise ValueError("BATADAL split ratios produced an empty split; check dataset size and ratios")

    return {
        "train": dataset.iloc[:train_end].reset_index(drop=True),
        "validation": dataset.iloc[train_end:validation_end].reset_index(drop=True),
        "test": dataset.iloc[validation_end:].reset_index(drop=True),
    }


def split_features_and_target(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [column for column in feature_columns + [target_column] if column not in dataset.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    features = dataset.loc[:, feature_columns].copy()
    target = dataset.loc[:, target_column].copy()
    return features, target


def split_skab_by_group_holdout(
    dataset: pd.DataFrame,
    group_column: str,
    target_column: str,
    split_config: dict,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create train/validation/test splits with disjoint source files."""
    required_columns = {group_column, target_column}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    train_ratio = split_config["train"]
    validation_ratio = split_config["validation"]
    test_ratio = split_config["test"]
    total_ratio = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"SKAB split ratios must sum to 1.0, got {total_ratio}")

    groups = sorted(dataset[group_column].unique().tolist())
    if len(groups) < 3:
        raise ValueError("SKAB dataset must contain at least 3 groups for holdout splitting")

    rng = np.random.default_rng(random_state)
    shuffled_groups = list(rng.permutation(groups))

    train_group_count = max(1, int(round(len(shuffled_groups) * train_ratio)))
    validation_group_count = max(1, int(round(len(shuffled_groups) * validation_ratio)))
    test_group_count = len(shuffled_groups) - train_group_count - validation_group_count

    if test_group_count <= 0:
        test_group_count = 1
        if train_group_count >= validation_group_count and train_group_count > 1:
            train_group_count -= 1
        elif validation_group_count > 1:
            validation_group_count -= 1
        else:
            raise ValueError("Not enough SKAB groups to create non-empty holdout splits")

    train_groups = set(shuffled_groups[:train_group_count])
    validation_groups = set(shuffled_groups[train_group_count : train_group_count + validation_group_count])
    test_groups = set(shuffled_groups[train_group_count + validation_group_count :])

    if not train_groups or not validation_groups or not test_groups:
        raise ValueError("SKAB holdout split produced an empty partition")

    return {
        "train": dataset[dataset[group_column].isin(train_groups)].reset_index(drop=True),
        "validation": dataset[dataset[group_column].isin(validation_groups)].reset_index(drop=True),
        "test": dataset[dataset[group_column].isin(test_groups)].reset_index(drop=True),
    }
