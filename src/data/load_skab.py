from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _resolve_skab_root(raw_data_path: str | Path) -> Path:
    root = Path(raw_data_path)
    if root.name.lower() == "skab":
        return root
    return root / "SKAB"


def _iter_csv_files(folder: Path) -> Iterable[Path]:
    return sorted(path for path in folder.rglob("*.csv") if path.is_file())


def load_skab_dataset(dataset_config: dict, raw_data_path: str | Path) -> pd.DataFrame:
    """Load and concatenate the required SKAB valve folders."""
    skab_root = _resolve_skab_root(raw_data_path)
    valve_folders = dataset_config.get("valve_folders", [])

    if not skab_root.exists():
        raise FileNotFoundError(f"SKAB directory not found: {skab_root}")

    frames: list[pd.DataFrame] = []
    for valve_folder in valve_folders:
        valve_path = skab_root / valve_folder
        if not valve_path.exists():
            raise FileNotFoundError(f"SKAB valve folder not found: {valve_path}")

        csv_files = list(_iter_csv_files(valve_path))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under: {valve_path}")

        for csv_path in csv_files:
            frame = pd.read_csv(csv_path)
            frame["source_group"] = valve_folder
            frame["source_file"] = csv_path.stem
            frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    if "datetime" in dataset.columns:
        dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce")
        dataset = dataset.sort_values(["source_file", "datetime"], kind="stable")

    target_column = dataset_config["target_column"]
    if target_column not in dataset.columns:
        raise KeyError(f"Target column '{target_column}' not found in SKAB dataset")

    return dataset.reset_index(drop=True)


def get_skab_feature_columns(dataset: pd.DataFrame, dataset_config: dict) -> list[str]:
    drop_columns = set(dataset_config.get("drop_columns", []))
    target_column = dataset_config["target_column"]
    return [column for column in dataset.columns if column not in drop_columns and column != target_column]
