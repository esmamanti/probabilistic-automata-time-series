from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.load_batadal import get_batadal_feature_columns, load_batadal_dataset
from data.load_skab import get_skab_feature_columns, load_skab_dataset
from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from data.split import generate_skab_group_folds, split_batadal_by_time, split_features_and_target
from utils.config import load_config


def summarize_skab(config: dict) -> None:
    raw_data_path = config["paths"]["raw_data"]
    dataset_config = config["datasets"]["skab"]

    dataset = load_skab_dataset(dataset_config, raw_data_path)
    feature_columns = get_skab_feature_columns(dataset, dataset_config)
    target_column = dataset_config["target_column"]
    group_column = dataset_config["group_column"]
    group_count = dataset[group_column].nunique()
    fold_count = min(5, group_count)

    print("=== SKAB ===")
    print(f"shape: {dataset.shape}")
    print(f"feature_count: {len(feature_columns)}")
    print(f"group_count: {group_count}")
    print(f"label_distribution: {dataset[target_column].value_counts().to_dict()}")

    if fold_count >= 2:
        first_fold = next(
            generate_skab_group_folds(
                dataset,
                group_column=group_column,
                target_column=target_column,
                n_splits=fold_count,
                random_state=config["project"]["random_seeds"][0],
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
    preprocessing_config = config["preprocessing"]

    dataset = load_batadal_dataset(dataset_config, raw_data_path)
    feature_columns = get_batadal_feature_columns(dataset, dataset_config)
    target_column = dataset_config["target_column"]
    splits = split_batadal_by_time(dataset, dataset_config["split"])

    x_train, y_train = split_features_and_target(splits["train"], feature_columns, target_column)
    x_validation, y_validation = split_features_and_target(splits["validation"], feature_columns, target_column)
    x_test, y_test = split_features_and_target(splits["test"], feature_columns, target_column)

    pipeline = PreprocessingPipeline(preprocessing_config)
    transformed_train = pipeline.fit_transform(x_train)
    transformed_validation = pipeline.transform(x_validation)
    transformed_test = pipeline.transform(x_test)

    print("=== BATADAL ===")
    print(f"shape: {dataset.shape}")
    print(f"feature_count: {len(feature_columns)}")
    print(f"label_distribution: {dataset[target_column].value_counts().to_dict()}")
    print(f"split_sizes: {{'train': {len(x_train)}, 'validation': {len(x_validation)}, 'test': {len(x_test)}}}")
    print(f"train_label_distribution: {y_train.value_counts().to_dict()}")
    print(f"validation_label_distribution: {y_validation.value_counts().to_dict()}")
    print(f"test_label_distribution: {y_test.value_counts().to_dict()}")
    print(
        "transformed_shapes: "
        f"{{'train': {transformed_train.shape}, 'validation': {transformed_validation.shape}, 'test': {transformed_test.shape}}}"
    )
    print()


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    summarize_skab(config)
    summarize_batadal(config)


if __name__ == "__main__":
    main()
