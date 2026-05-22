from __future__ import annotations

from pathlib import Path

import pandas as pd


def _resolve_batadal_root(raw_data_path: str | Path) -> Path:
    root = Path(raw_data_path)
    if root.name.lower() == "batadal":
        return root
    return root / "BATADAL"


def load_batadal_dataset(dataset_config: dict, raw_data_path: str | Path) -> pd.DataFrame:
    """Load BATADAL Training Dataset 2 and preserve time order."""
    batadal_root = _resolve_batadal_root(raw_data_path)
    if not batadal_root.exists():
        raise FileNotFoundError(f"BATADAL directory not found: {batadal_root}")

    csv_path = batadal_root / dataset_config["train_file"]
    if not csv_path.exists():
        raise FileNotFoundError(f"BATADAL file not found: {csv_path}")

    dataset = pd.read_csv(csv_path)

    datetime_candidates = ["DATETIME", "datetime", "DateTime", "Timestamp"]
    datetime_column = next((column for column in datetime_candidates if column in dataset.columns), None)
    if datetime_column is not None:
        dataset[datetime_column] = pd.to_datetime(dataset[datetime_column], errors="coerce")
        dataset = dataset.sort_values(datetime_column, kind="stable")

    target_column = dataset_config["target_column"]
    if target_column not in dataset.columns:
        raise KeyError(f"Target column '{target_column}' not found in BATADAL dataset")

    return dataset.reset_index(drop=True)


def get_batadal_feature_columns(dataset: pd.DataFrame, dataset_config: dict) -> list[str]:
    drop_columns = set(dataset_config.get("drop_columns", []))
    target_column = dataset_config["target_column"]
    return [column for column in dataset.columns if column not in drop_columns and column != target_column]
