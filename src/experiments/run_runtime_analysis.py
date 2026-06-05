from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.metrics import aggregate_metrics_frame
from evaluation.plots import save_figure
from utils.config import load_config


def ensure_output_dir(config: dict) -> Path:
    output_dir = PROJECT_ROOT / config["paths"]["runtime_results"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_runtime_frames() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in (
        PROJECT_ROOT / "results" / "tables" / "automata_runtime_metrics.csv",
        PROJECT_ROOT / "results" / "tables" / "deep_learning_runtime_metrics.csv",
    ):
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError("Runtime metrics were not found. Run baseline experiments first.")
    return pd.concat(frames, ignore_index=True)


def plot_runtime_comparison(runtime_df: pd.DataFrame) -> plt.Figure:
    summary_df = (
        runtime_df.groupby("model", dropna=False)[["training_time_seconds", "inference_time_seconds"]]
        .mean()
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(summary_df["model"].astype(str), summary_df["training_time_seconds"].astype(float))
    axes[0].set_title("Training Time")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(summary_df["model"].astype(str), summary_df["inference_time_seconds"].astype(float), color="#4c78a8")
    axes[1].set_title("Inference Time")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    if not bool(config.get("runtime_analysis", {}).get("enabled", True)):
        print("Runtime analysis disabled in config.")
        return

    output_dir = ensure_output_dir(config)
    runtime_df = load_runtime_frames()
    summary_df = aggregate_metrics_frame(
        runtime_df,
        group_columns=["dataset", "family", "model", "split"],
        metric_columns=["training_time_seconds", "inference_time_seconds", "test_examples"],
    )
    summary_df.to_csv(output_dir / "runtime_comparison.csv", index=False)
    figure = plot_runtime_comparison(runtime_df)
    save_figure(figure, output_dir / "runtime_comparison.png")

    print("=== Runtime Analysis Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
