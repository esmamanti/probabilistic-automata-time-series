import pandas as pd

from data.split import split_skab_by_group_holdout


def test_skab_holdout_split_keeps_source_files_disjoint():
    dataset = pd.DataFrame(
        {
            "source_file": ["f1", "f1", "f2", "f2", "f3", "f3", "f4", "f4", "f5", "f5"],
            "anomaly": [0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
            "sensor": range(10),
        }
    )

    splits = split_skab_by_group_holdout(
        dataset=dataset,
        group_column="source_file",
        target_column="anomaly",
        split_config={"train": 0.6, "validation": 0.2, "test": 0.2},
        random_state=42,
    )

    train_groups = set(splits["train"]["source_file"])
    validation_groups = set(splits["validation"]["source_file"])
    test_groups = set(splits["test"]["source_file"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
