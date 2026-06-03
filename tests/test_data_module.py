import pandas as pd

from data.data_module import DataModule


def test_prepare_skab_fold_datasets_uses_group_folds_without_leakage(monkeypatch):
    dataset = pd.DataFrame(
        {
            "source_file": ["f1", "f1", "f2", "f2", "f3", "f3", "f4", "f4", "f5", "f5"],
            "source_group": ["valve1"] * 10,
            "datetime": pd.date_range("2026-01-01", periods=10, freq="h"),
            "changepoint": [0] * 10,
            "anomaly": [0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
            "sensor": [0.1, 0.2, 0.5, 0.7, 1.1, 1.2, 0.3, 0.4, 0.8, 0.9],
        }
    )

    monkeypatch.setattr("data.data_module.load_skab_dataset", lambda dataset_config, raw_data_path: dataset.copy())
    monkeypatch.setattr("data.data_module.get_skab_feature_columns", lambda dataset, dataset_config: ["sensor"])

    config = {
        "paths": {"raw_data": "data/raw"},
        "datasets": {
            "skab": {
                "target_column": "anomaly",
                "group_column": "source_file",
                "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
            }
        },
        "preprocessing": {
            "scaler": "standard",
            "pca": {"enabled": False},
            "sequence_length": 1,
            "sequence_stride": 1,
        },
        "project": {"random_seeds": [42]},
        "noise": {"gaussian_mean": 0.0, "gaussian_std": 0.05},
    }

    prepared_folds = DataModule(config).prepare_skab_fold_datasets()

    assert len(prepared_folds) == 5
    for fold_index, prepared in enumerate(prepared_folds):
        train_groups = set(prepared.splits["train"].frame["source_file"])
        validation_groups = set(prepared.splits["validation"].frame["source_file"])
        test_groups = set(prepared.splits["test"].frame["source_file"])

        assert prepared.evaluation_split == f"fold_{fold_index}"
        assert train_groups
        assert validation_groups
        assert test_groups
        assert train_groups.isdisjoint(validation_groups)
        assert train_groups.isdisjoint(test_groups)
        assert validation_groups.isdisjoint(test_groups)
