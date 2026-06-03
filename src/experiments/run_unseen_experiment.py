from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule, PreparedDataset
from evaluation.metrics import aggregate_metrics_frame
from experiments.run_automata import (
    build_automata_model,
    build_explanation_frame,
    calibrate_threshold,
    derive_pattern_labels,
    extract_1d_series,
    extract_pattern_scores,
    get_decision_config,
)
from experiments.run_deep_models import build_model, build_trainer
from utils.config import load_config
from utils.seed import clone_config_with_seed, get_experiment_seeds, set_seed


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def derive_pattern_end_indices(
    total_rows: int,
    paa_window_size: int,
    pattern_window_size: int,
    stride: int,
    pattern_count: int,
) -> list[int]:
    end_indices: list[int] = []
    for pattern_index in range(pattern_count):
        start = pattern_index * stride * paa_window_size
        end = min(start + (pattern_window_size * paa_window_size), total_rows)
        end_indices.append(max(0, end - 1))
    return end_indices


def compute_subset_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    if predictions_df.empty:
        nan_value = float("nan")
        return {
            "accuracy": nan_value,
            "precision": nan_value,
            "recall": nan_value,
            "f1_score": nan_value,
            "anomaly_rate": nan_value,
            "true_anomaly_rate": nan_value,
        }

    y_true = predictions_df["true_label"].astype(int)
    y_pred = predictions_df["predicted_label"].astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    predicted_positive = int((y_pred == 1).sum())
    actual_positive = int((y_true == 1).sum())
    accuracy = float((y_true == y_pred).mean())
    precision = float(true_positive / predicted_positive) if predicted_positive else 0.0
    recall = float(true_positive / actual_positive) if actual_positive else 0.0
    f1_score = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "anomaly_rate": float((y_pred == 1).mean()),
        "true_anomaly_rate": float((y_true == 1).mean()),
    }


def summarize_unseen_subset(
    predictions_df: pd.DataFrame,
    *,
    dataset_name: str,
    seed: int,
    split_name: str,
    model_name: str,
    family: str,
) -> dict[str, object]:
    metrics = compute_subset_metrics(predictions_df)
    summary_row: dict[str, object] = {
        "dataset": dataset_name.upper(),
        "seed": int(seed),
        "split": split_name,
        "model": model_name,
        "family": family,
        "unseen_examples": int(len(predictions_df)),
        **metrics,
    }

    if "distance" in predictions_df.columns:
        unseen_distances = predictions_df["distance"].dropna()
        summary_row["avg_unseen_distance"] = float(unseen_distances.mean()) if not unseen_distances.empty else math.nan
    if "confidence_score" in predictions_df.columns:
        unseen_confidences = predictions_df["confidence_score"].dropna()
        summary_row["avg_unseen_confidence"] = float(unseen_confidences.mean()) if not unseen_confidences.empty else math.nan

    return summary_row


def build_automata_unseen_reference(
    prepared_dataset: PreparedDataset,
    *,
    dataset_name: str,
    seed: int,
    models_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_name = prepared_dataset.evaluation_split or "test"
    model = build_automata_model(models_config)
    decision_config = get_decision_config(models_config)
    automata_config = models_config["automata"]

    train_series = extract_1d_series(prepared_dataset.splits["train"].features)
    validation_series = extract_1d_series(prepared_dataset.splits["validation"].features)
    test_series = extract_1d_series(prepared_dataset.splits["test"].features)
    model.fit(train_series)

    calibration_score_result = model.score_sequence(validation_series)
    calibration_labels = derive_pattern_labels(
        raw_labels=prepared_dataset.splits["validation"].target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(calibration_score_result["explanations"]),
    )
    threshold = calibrate_threshold(
        scores=extract_pattern_scores(calibration_score_result, str(decision_config["score_field"])),
        labels=calibration_labels,
        fallback_quantile=float(decision_config["fallback_quantile"]),
    )

    score_result = model.score_sequence(test_series)
    true_labels = derive_pattern_labels(
        raw_labels=prepared_dataset.splits["test"].target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(score_result["explanations"]),
    )
    explanations_df = build_explanation_frame(
        dataset_name=dataset_name.upper(),
        split_name=split_name,
        score_result=score_result,
        true_labels=true_labels,
        score_field=str(decision_config["score_field"]),
        threshold=threshold,
    )
    explanations_df["row_index"] = derive_pattern_end_indices(
        total_rows=len(prepared_dataset.splits["test"].target),
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(explanations_df),
    )
    explanations_df["seed"] = int(seed)
    explanations_df["model"] = "AUTOMATA"
    explanations_df["family"] = "AUTOMATA"
    explanations_df["predicted_probability"] = pd.Series([math.nan] * len(explanations_df), dtype=float)
    explanations_df["reference_source"] = "automata"

    unseen_df = explanations_df[explanations_df["status"] == "unseen"].reset_index(drop=True)
    return explanations_df, unseen_df


def build_deep_unseen_predictions(
    prepared_dataset: PreparedDataset,
    unseen_reference_df: pd.DataFrame,
    *,
    dataset_name: str,
    seed: int,
    config: dict,
    models_config: dict,
) -> list[tuple[str, pd.DataFrame]]:
    if unseen_reference_df.empty:
        return [(model_name.upper(), pd.DataFrame()) for model_name in ("lstm", "gru")]

    split_name = prepared_dataset.evaluation_split or "test"
    unseen_reference = unseen_reference_df.loc[
        :,
        [
            "row_index",
            "pattern",
            "mapped_to",
            "distance",
            "confidence_score",
            "decision_reason",
            "status",
        ],
    ].drop_duplicates(subset=["row_index"])

    prediction_frames: list[tuple[str, pd.DataFrame]] = []
    sequence_data = prepared_dataset.splits["test"].sequences

    for model_name in ("lstm", "gru"):
        set_seed(int(seed))
        model = build_model(model_name, models_config["deep_learning"][model_name])
        trainer = build_trainer(model_name, model, config, models_config)
        trainer.fit(
            train_data=prepared_dataset.splits["train"].sequences,
            validation_data=prepared_dataset.splits["validation"].sequences,
        )
        probabilities = trainer.predict_probabilities(sequence_data)
        predictions = trainer.predict_labels(sequence_data)
        frame = pd.DataFrame(
            {
                "dataset": dataset_name.upper(),
                "seed": int(seed),
                "split": split_name,
                "model": model_name.upper(),
                "family": "DEEP",
                "row_index": sequence_data.sequence_end_indices.astype(int),
                "true_label": sequence_data.targets.astype(int),
                "predicted_label": predictions.astype(int),
                "predicted_probability": probabilities.astype(float),
            }
        )
        frame = frame.merge(unseen_reference, on="row_index", how="inner")
        if not frame.empty:
            frame["reference_source"] = "automata"
            frame["decision"] = frame["predicted_label"].map({0: "normal", 1: "anomaly"})
        prediction_frames.append((model_name.upper(), frame))

    return prediction_frames


def analyze_unseen_for_dataset(dataset_name: str, config: dict, models_config: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_module = DataModule(config)
    prepared_datasets = (
        data_module.prepare_skab_fold_datasets(scenario="original")
        if dataset_name.lower() == "skab"
        else [data_module.prepare_dataset(dataset_name, scenario="original")]
    )

    explanation_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for prepared_dataset in prepared_datasets:
        split_name = prepared_dataset.evaluation_split or "test"
        _, automata_unseen_df = build_automata_unseen_reference(
            prepared_dataset,
            dataset_name=dataset_name,
            seed=seed,
            models_config=models_config,
        )

        explanation_frames.append(automata_unseen_df)
        summary_rows.append(
            summarize_unseen_subset(
                automata_unseen_df,
                dataset_name=dataset_name,
                seed=seed,
                split_name=split_name,
                model_name="AUTOMATA",
                family="AUTOMATA",
            )
        )

        for model_name, deep_frame in build_deep_unseen_predictions(
            prepared_dataset,
            automata_unseen_df,
            dataset_name=dataset_name,
            seed=seed,
            config=config,
            models_config=models_config,
        ):
            if not deep_frame.empty:
                explanation_frames.append(deep_frame)
            summary_rows.append(
                summarize_unseen_subset(
                    deep_frame,
                    dataset_name=dataset_name,
                    seed=seed,
                    split_name=split_name,
                    model_name=model_name,
                    family="DEEP",
                )
            )

    explanations_df = pd.concat(explanation_frames, ignore_index=True) if explanation_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return explanations_df, summary_df


def build_aggregated_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "anomaly_rate", "true_anomaly_rate"]
    available_metric_columns = [column for column in metric_columns if column in summary_df.columns]
    aggregated = aggregate_metrics_frame(
        summary_df,
        group_columns=["dataset", "model", "family"],
        metric_columns=available_metric_columns,
    )
    unseen_counts = (
        summary_df.groupby(["dataset", "model", "family"], dropna=False)["unseen_examples"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "unseen_examples_mean",
                "std": "unseen_examples_std",
                "min": "unseen_examples_min",
                "max": "unseen_examples_max",
            }
        )
    )
    return aggregated.merge(unseen_counts, on=["dataset", "model", "family"], how="left")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)
    explanation_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        skab_explanations, skab_summary = analyze_unseen_for_dataset("skab", seed_config, models_config, int(seed))
        batadal_explanations, batadal_summary = analyze_unseen_for_dataset("batadal", seed_config, models_config, int(seed))
        explanation_frames.extend([skab_explanations, batadal_explanations])
        summary_frames.extend([skab_summary, batadal_summary])

    explanations_df = pd.concat(explanation_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    aggregated_summary_df = build_aggregated_summary(summary_df)

    explanations_df.to_csv(explanations_dir / "unseen_explanations.csv", index=False)
    summary_df.to_csv(tables_dir / "unseen_metrics.csv", index=False)
    aggregated_summary_df.to_csv(tables_dir / "unseen_metrics_summary.csv", index=False)
    with (explanations_dir / "unseen_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": summary_df.to_dict(orient="records"),
                "summary": aggregated_summary_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    print("=== Unseen Experiment Summary ===")
    print(summary_df.to_string(index=False))
    print()
    print("=== Unseen Aggregated Summary ===")
    print(aggregated_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
