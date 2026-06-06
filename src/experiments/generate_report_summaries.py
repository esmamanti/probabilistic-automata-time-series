from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import load_config


def ensure_tables_dir(config: dict) -> Path:
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def build_model_prf_summary(tables_dir: Path) -> Path:
    summary_df = load_csv(tables_dir / "model_comparison_metrics_summary.csv")
    metric_columns = ["accuracy_mean", "precision_mean", "recall_mean", "f1_score_mean"]
    report_df = (
        summary_df.loc[:, ["dataset", "model", *metric_columns]]
        .groupby(["dataset", "model"], dropna=False)
        .mean()
        .reset_index()
        .sort_values(["dataset", "f1_score_mean"], ascending=[True, False], kind="stable")
    )
    output_path = tables_dir / "report_model_prf_summary.csv"
    report_df.to_csv(output_path, index=False)
    return output_path


def build_batadal_threshold_summary(tables_dir: Path) -> Path:
    threshold_df = load_csv(PROJECT_ROOT / "results" / "thresholds" / "threshold_tuning_results.csv")
    batadal_df = threshold_df[threshold_df["dataset"].astype(str).str.upper() == "BATADAL"].copy()
    if batadal_df.empty:
        raise ValueError("Threshold tuning results do not contain BATADAL rows")

    summary_df = (
        batadal_df.groupby("model", dropna=False)[
            ["best_threshold", "best_val_precision", "best_val_recall", "best_val_f1", "best_score"]
        ]
        .agg(["mean", "std", "min", "max"])
    )
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index().sort_values("best_val_f1_mean", ascending=False, kind="stable")
    output_path = tables_dir / "batadal_threshold_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def build_runtime_speedup_summary(tables_dir: Path) -> Path:
    runtime_df = load_csv(PROJECT_ROOT / "results" / "runtime" / "runtime_comparison.csv")
    summary_df = (
        runtime_df.groupby(["dataset", "model"], dropna=False)[
            ["training_time_seconds_mean", "inference_time_seconds_mean"]
        ]
        .mean()
        .reset_index()
    )
    automata_df = (
        summary_df[summary_df["model"].astype(str).str.upper() == "AUTOMATA"]
        .rename(
            columns={
                "training_time_seconds_mean": "automata_training_time_seconds",
                "inference_time_seconds_mean": "automata_inference_time_seconds",
            }
        )
        .loc[:, ["dataset", "automata_training_time_seconds", "automata_inference_time_seconds"]]
    )
    comparison_df = summary_df.merge(automata_df, on="dataset", how="left")
    comparison_df["training_speedup_vs_automata"] = (
        comparison_df["training_time_seconds_mean"] / comparison_df["automata_training_time_seconds"]
    )
    comparison_df["inference_speedup_vs_automata"] = (
        comparison_df["inference_time_seconds_mean"] / comparison_df["automata_inference_time_seconds"]
    )
    comparison_df = comparison_df.sort_values(["dataset", "model"], kind="stable")
    output_path = tables_dir / "runtime_speedup_summary.csv"
    comparison_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    tables_dir = ensure_tables_dir(config)

    generated_paths = [
        build_model_prf_summary(tables_dir),
        build_batadal_threshold_summary(tables_dir),
        build_runtime_speedup_summary(tables_dir),
    ]

    print("=== Generated Report Summaries ===")
    for path in generated_paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
