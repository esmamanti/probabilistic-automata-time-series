from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule
from data.load_batadal import load_batadal_dataset
from data.load_skab import load_skab_dataset
from data.split import generate_skab_group_folds
from utils.config import load_config
from utils.seed import get_primary_seed


def summarize_skab(config: dict) -> None:
    raw_data_path = config["paths"]["raw_data"]
    dataset_config = config["datasets"]["skab"]
    data_module = DataModule(config)

    dataset = load_skab_dataset(dataset_config, raw_data_path)
    target_column = dataset_config["target_column"]
    group_column = dataset_config["group_column"]
    group_count = dataset[group_column].nunique()
    fold_count = min(5, group_count)
    prepared_folds = data_module.prepare_skab_fold_datasets()
    first_fold_dataset = prepared_folds[0] if prepared_folds else None

    print("=== SKAB ===")
    print(f"shape: {dataset.shape}")
    print(f"feature_count: {len(first_fold_dataset.feature_columns) if first_fold_dataset is not None else 0}")
    print(f"group_count: {group_count}")
    print(f"fold_count: {len(prepared_folds)}")
    print(f"label_distribution: {dataset[target_column].value_counts().to_dict()}")
    if first_fold_dataset is not None:
        print(
            "first_fold_split_sizes: "
            f"{{'train': {len(first_fold_dataset.splits['train'].frame)}, "
            f"'validation': {len(first_fold_dataset.splits['validation'].frame)}, "
            f"'test': {len(first_fold_dataset.splits['test'].frame)}}}"
        )
        print(
            "first_fold_sequence_shapes: "
            f"{{'train': {first_fold_dataset.splits['train'].sequences.features.shape}, "
            f"'validation': {first_fold_dataset.splits['validation'].sequences.features.shape}, "
            f"'test': {first_fold_dataset.splits['test'].sequences.features.shape}}}"
        )

    if fold_count >= 2:
        first_fold = next(
            generate_skab_group_folds(
                dataset,
                group_column=group_column,
                target_column=target_column,
                n_splits=fold_count,
                random_state=get_primary_seed(config),
            )
        )
        _, train_idx, test_idx = first_fold
        print(f"first_fold_train_rows: {len(train_idx)}")
        print(f"first_fold_test_rows: {len(test_idx)}")
    else:
        print("first_fold_train_rows: not enough groups for GroupKFold")
        print("first_fold_test_rows: not enough groups for GroupKFold")

    print()


def summarize_batadal(config: dict) -> None:
    raw_data_path = config["paths"]["raw_data"]
    dataset_config = config["datasets"]["batadal"]
    data_module = DataModule(config)

    dataset = load_batadal_dataset(dataset_config, raw_data_path)
    target_column = dataset_config["target_column"]
    prepared = data_module.prepare_dataset("batadal")
    y_train = prepared.splits["train"].target
    y_validation = prepared.splits["validation"].target
    y_test = prepared.splits["test"].target

    print("=== BATADAL ===")
    print(f"shape: {dataset.shape}")
    print(f"feature_count: {len(prepared.feature_columns)}")
    print(f"label_distribution: {dataset[target_column].value_counts().to_dict()}")
    print(
        "split_sizes: "
        f"{{'train': {len(prepared.splits['train'].frame)}, "
        f"'validation': {len(prepared.splits['validation'].frame)}, "
        f"'test': {len(prepared.splits['test'].frame)}}}"
    )
    print(f"train_label_distribution: {y_train.value_counts().to_dict()}")
    print(f"validation_label_distribution: {y_validation.value_counts().to_dict()}")
    print(f"test_label_distribution: {y_test.value_counts().to_dict()}")
    print(
        "transformed_shapes: "
        f"{{'train': {prepared.splits['train'].features.shape}, "
        f"'validation': {prepared.splits['validation'].features.shape}, "
        f"'test': {prepared.splits['test'].features.shape}}}"
    )
    print(
        "sequence_shapes: "
        f"{{'train': {prepared.splits['train'].sequences.features.shape}, "
        f"'validation': {prepared.splits['validation'].sequences.features.shape}, "
        f"'test': {prepared.splits['test'].sequences.features.shape}}}"
    )
    print()


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    summarize_skab(config)
    summarize_batadal(config)


if __name__ == "__main__":
    main()
