from __future__ import annotations

import json
import sys
from time import perf_counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule, PreparedDataset
from evaluation.evaluator import Evaluator
from evaluation.metrics import aggregate_metrics_frame
from evaluation.plots import save_figure
from experiments.run_automata import run_batadal_experiment, run_skab_experiment
from models.deep_learning.cnn_model import CNNModel
from models.deep_learning.gru_model import GRUModel
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.trainer import Trainer, tune_decision_threshold
from utils.config import load_config
from utils.experiment_context import attach_context_to_record, build_run_context
from utils.seed import clone_config_with_seed, get_primary_seed, set_seed


def ensure_output_dirs(config: dict) -> tuple[Path, Path, Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    thresholds_dir = PROJECT_ROOT / config["paths"]["thresholds"]
    improvements_dir = PROJECT_ROOT / config["paths"]["improvements"]
    for path in (explanations_dir, tables_dir, thresholds_dir, improvements_dir):
        path.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir, thresholds_dir, improvements_dir


def resolve_device(config: dict) -> str:
    configured_device = str(config["project"].get("device", "cpu")).lower()
    if configured_device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def get_enabled_deep_models(models_config: dict) -> list[str]:
    enabled_models: list[str] = []
    for model_name, model_config in models_config.get("deep_learning", {}).items():
        if bool(model_config.get("enabled", True)):
            enabled_models.append(model_name.lower())
    if not enabled_models:
        raise ValueError("No enabled deep learning models found under models_config['deep_learning']")
    return enabled_models


def get_primary_deep_model_name(models_config: dict) -> str:
    return get_enabled_deep_models(models_config)[0].upper()


def build_model(model_name: str, model_config: dict):
    architecture = str(model_config.get("architecture", model_name)).lower()
    if architecture == "lstm":
        return LSTMModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            output_size=model_config.get("output_size", 1),
        )
    if architecture == "gru":
        return GRUModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            output_size=model_config.get("output_size", 1),
        )
    if architecture == "cnn":
        return CNNModel(
            input_channels=model_config["input_channels"],
            num_filters=model_config["num_filters"],
            kernel_size=model_config["kernel_size"],
            dropout=model_config["dropout"],
            output_size=model_config.get("output_size", 1),
        )
    raise ValueError(f"Unsupported deep learning model architecture: {architecture}")


def build_trainer(dataset_name: str, model_name: str, model, config: dict, models_config: dict) -> Trainer:
    training_config = models_config["training"]
    model_config = models_config["deep_learning"][model_name]
    class_imbalance_config = training_config.get("class_imbalance", {})
    dataset_key = str(dataset_name).upper()
    model_key = str(model_name).upper()
    loss_config = {
        "loss_name": str(class_imbalance_config.get("loss_name", "bce")),
        "focal_gamma": float(class_imbalance_config.get("focal_gamma", 2.0)),
    }
    for override in class_imbalance_config.get("dataset_model_overrides", []):
        if str(override.get("dataset", "")).upper() == dataset_key and str(override.get("model", "")).upper() == model_key:
            loss_config.update(
                {
                    "loss_name": str(override.get("loss_name", loss_config["loss_name"])),
                    "focal_gamma": float(override.get("focal_gamma", loss_config["focal_gamma"])),
                }
            )
    early_stopping_config = training_config["early_stopping"]
    return Trainer(
        model=model,
        learning_rate=float(model_config["learning_rate"]),
        batch_size=int(training_config["batch_size"]),
        epochs=int(training_config["epochs"]),
        device=resolve_device(config),
        early_stopping_enabled=bool(early_stopping_config["enabled"]),
        early_stopping_patience=int(early_stopping_config["patience"]),
        early_stopping_monitor=str(early_stopping_config.get("monitor", "val_loss")),
        early_stopping_mode=str(early_stopping_config.get("mode", "min")),
        use_pos_weight=bool(class_imbalance_config.get("use_pos_weight", False)),
        pos_weight_strategy=str(class_imbalance_config.get("pos_weight_strategy", "neg_pos_ratio")),
        loss_name=loss_config["loss_name"],
        focal_gamma=float(loss_config["focal_gamma"]),
    )


def build_prediction_frame(
    dataset_name: str,
    model_name: str,
    split_name: str,
    prepared_dataset: PreparedDataset,
    probabilities,
    predictions,
    seed: int,
    *,
    version: str,
    threshold: float,
) -> pd.DataFrame:
    sequence_data = prepared_dataset.splits["test"].sequences
    frame = prepared_dataset.splits["test"].frame.iloc[sequence_data.sequence_end_indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": model_name,
            "split": split_name,
            "seed": int(seed),
            "version": version,
            "threshold": float(threshold),
            "row_index": sequence_data.sequence_end_indices.astype(int),
            "true_label": sequence_data.targets.astype(int),
            "predicted_label": predictions.astype(int),
            "predicted_probability": probabilities.astype(float),
        }
    ).join(frame, how="left")


def compute_metrics_from_predictions(targets, predictions) -> dict[str, float]:
    targets_array = np.asarray(targets, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    return {
        "accuracy": float(accuracy_score(targets_array, predictions_array)),
        "precision": float(precision_score(targets_array, predictions_array, zero_division=0)),
        "recall": float(recall_score(targets_array, predictions_array, zero_division=0)),
        "f1_score": float(f1_score(targets_array, predictions_array, zero_division=0)),
    }


def build_threshold_tuning_frame(
    *,
    dataset_name: str,
    model_name: str,
    split_name: str,
    seed: int,
    threshold_result: dict[str, float],
) -> dict[str, object]:
    return {
        "dataset": dataset_name.upper(),
        "model": model_name.upper(),
        "split": split_name,
        "seed": int(seed),
        "metric": str(threshold_result.get("metric", "f1")),
        "beta": float(threshold_result.get("beta", 2.0)),
        "best_threshold": float(threshold_result["threshold"]),
        "min_threshold": float(threshold_result.get("min_threshold", 0.01)),
        "best_score": float(threshold_result.get("score", threshold_result["f1_score"])),
        "best_val_f1": float(threshold_result["f1_score"]),
        "best_val_precision": float(threshold_result["precision"]),
        "best_val_recall": float(threshold_result["recall"]),
    }


def build_probability_distribution_frame(
    *,
    dataset_name: str,
    model_name: str,
    split_name: str,
    seed: int,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset_name.upper(),
            "model": model_name.upper(),
            "split": split_name,
            "seed": int(seed),
            "true_label": np.asarray(true_labels, dtype=int),
            "predicted_probability": np.asarray(probabilities, dtype=float),
        }
    )


def resolve_threshold_tuning_config(dataset_name: str, models_config: dict) -> dict[str, float | str | bool]:
    training_config = models_config.get("training", {})
    threshold_tuning_config = training_config.get("threshold_tuning", {})
    dataset_override = threshold_tuning_config.get("dataset_overrides", {}).get(dataset_name.upper(), {})
    merged_config = {
        "enabled": bool(threshold_tuning_config.get("enabled", True)),
        "metric": str(threshold_tuning_config.get("metric", "f1")),
        "beta": float(threshold_tuning_config.get("beta", 2.0)),
        "start": float(threshold_tuning_config.get("start", 0.01)),
        "end": float(threshold_tuning_config.get("end", 0.99)),
        "step": float(threshold_tuning_config.get("step", 0.01)),
        "min_threshold": float(threshold_tuning_config.get("min_threshold", threshold_tuning_config.get("start", 0.01))),
    }
    merged_config.update(dataset_override)
    if "beta" in merged_config:
        merged_config["beta"] = float(merged_config["beta"])
    return merged_config


def tune_threshold_for_validation(dataset_name: str, sequence_data, trainer: Trainer, models_config: dict) -> dict[str, float]:
    probabilities = trainer.predict_probabilities(sequence_data)
    threshold_config = resolve_threshold_tuning_config(dataset_name, models_config)
    if not bool(threshold_config["enabled"]):
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "score": 0.0,
            "metric": str(threshold_config["metric"]),
            "beta": float(threshold_config["beta"]),
        }
    return tune_decision_threshold(
        probabilities,
        sequence_data.targets,
        metric=str(threshold_config["metric"]),
        beta=float(threshold_config["beta"]),
        start=float(threshold_config["start"]),
        end=float(threshold_config["end"]),
        step=float(threshold_config["step"]),
        min_threshold=float(threshold_config["min_threshold"]),
    )


def summarize_ratios(prepared_dataset: PreparedDataset) -> dict[str, float]:
    return {
        "anomaly_ratio_train": float(np.asarray(prepared_dataset.splits["train"].sequences.targets, dtype=float).mean()),
        "anomaly_ratio_val": float(np.asarray(prepared_dataset.splits["validation"].sequences.targets, dtype=float).mean()),
        "anomaly_ratio_test": float(np.asarray(prepared_dataset.splits["test"].sequences.targets, dtype=float).mean()),
    }


def train_and_evaluate_model(
    dataset_name: str,
    model_name: str,
    prepared_dataset: PreparedDataset,
    config: dict,
    models_config: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float], dict[str, object], dict[str, object], dict[str, object], pd.DataFrame]:
    split_name = prepared_dataset.evaluation_split or "test"
    model = build_model(model_name, models_config["deep_learning"][model_name])
    trainer = build_trainer(dataset_name, model_name, model, config, models_config)
    ratio_summary = summarize_ratios(prepared_dataset)
    training_started_at = perf_counter()
    history = trainer.fit(
        train_data=prepared_dataset.splits["train"].sequences,
        validation_data=prepared_dataset.splits["validation"].sequences,
    )
    training_time_seconds = perf_counter() - training_started_at

    threshold_result = tune_threshold_for_validation(dataset_name, prepared_dataset.splits["validation"].sequences, trainer, models_config)
    baseline_threshold = 0.5
    tuned_threshold = float(threshold_result["threshold"])

    inference_started_at = perf_counter()
    test_probabilities = trainer.predict_probabilities(prepared_dataset.splits["test"].sequences)
    inference_time_seconds = perf_counter() - inference_started_at

    baseline_predictions = (test_probabilities >= baseline_threshold).astype(int)
    tuned_predictions = (test_probabilities >= tuned_threshold).astype(int)

    baseline_metrics = compute_metrics_from_predictions(
        prepared_dataset.splits["test"].sequences.targets,
        baseline_predictions,
    )
    tuned_metrics = compute_metrics_from_predictions(
        prepared_dataset.splits["test"].sequences.targets,
        tuned_predictions,
    )
    probability_distribution_df = build_probability_distribution_frame(
        dataset_name=dataset_name,
        model_name=model_name,
        split_name=split_name,
        seed=seed,
        probabilities=test_probabilities,
        true_labels=prepared_dataset.splits["test"].sequences.targets,
    )

    context = build_run_context(
        config=config,
        models_config=models_config,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=int(seed),
        family="DEEP",
        model_name=model_name,
    )

    baseline_metrics_record = attach_context_to_record(
        {
            "dataset": dataset_name.upper(),
            "model": model_name.upper(),
            "split": split_name,
            "seed": int(seed),
            "version": "baseline_0.5_threshold",
            "threshold": float(baseline_threshold),
            "best_val_f1": float(threshold_result["f1_score"]),
            "best_val_precision": float(threshold_result["precision"]),
            "best_val_recall": float(threshold_result["recall"]),
            "epochs_completed": int(history.epochs_completed),
            "best_validation_loss": float(history.best_validation_loss),
            "best_monitor_name": history.best_monitor_name,
            "best_monitor_value": float(history.best_monitor_value),
            "pos_weight": float(history.pos_weight),
            "loss_name": history.loss_name,
            "test_examples": int(len(prepared_dataset.splits["test"].sequences.targets)),
            **ratio_summary,
            **baseline_metrics,
        },
        context,
    )
    tuned_metrics_record = attach_context_to_record(
        {
            "dataset": dataset_name.upper(),
            "model": model_name.upper(),
            "split": split_name,
            "seed": int(seed),
            "version": "tuned_threshold_weighted_loss",
            "threshold": float(tuned_threshold),
            "best_val_f1": float(threshold_result["f1_score"]),
            "best_val_precision": float(threshold_result["precision"]),
            "best_val_recall": float(threshold_result["recall"]),
            "epochs_completed": int(history.epochs_completed),
            "best_validation_loss": float(history.best_validation_loss),
            "best_monitor_name": history.best_monitor_name,
            "best_monitor_value": float(history.best_monitor_value),
            "pos_weight": float(history.pos_weight),
            "loss_name": history.loss_name,
            "test_examples": int(len(prepared_dataset.splits["test"].sequences.targets)),
            **ratio_summary,
            **tuned_metrics,
        },
        context,
    )

    baseline_predictions_df = build_prediction_frame(
        dataset_name=dataset_name.upper(),
        model_name=model_name.upper(),
        split_name=split_name,
        prepared_dataset=prepared_dataset,
        probabilities=test_probabilities,
        predictions=baseline_predictions,
        seed=seed,
        version="baseline_0.5_threshold",
        threshold=baseline_threshold,
    )
    tuned_predictions_df = build_prediction_frame(
        dataset_name=dataset_name.upper(),
        model_name=model_name.upper(),
        split_name=split_name,
        prepared_dataset=prepared_dataset,
        probabilities=test_probabilities,
        predictions=tuned_predictions,
        seed=seed,
        version="tuned_threshold_weighted_loss",
        threshold=tuned_threshold,
    )

    runtime_record = {
        "dataset": dataset_name.upper(),
        "model": model_name.upper(),
        "family": "DEEP",
        "split": split_name,
        "seed": int(seed),
        "training_time_seconds": float(training_time_seconds),
        "inference_time_seconds": float(inference_time_seconds),
        "test_examples": int(len(prepared_dataset.splits["test"].sequences.targets)),
        "epochs_completed": int(history.epochs_completed),
        "pos_weight": float(history.pos_weight),
        "loss_name": history.loss_name,
    }
    threshold_record = build_threshold_tuning_frame(
        dataset_name=dataset_name,
        model_name=model_name,
        split_name=split_name,
        seed=seed,
        threshold_result=threshold_result,
    )
    improvement_record = {
        "dataset": dataset_name.upper(),
        "model": model_name.upper(),
        "version": "baseline_0.5_threshold",
        "accuracy": float(baseline_metrics["accuracy"]),
        "precision": float(baseline_metrics["precision"]),
        "recall": float(baseline_metrics["recall"]),
        "f1_score": float(baseline_metrics["f1_score"]),
        "threshold": float(baseline_threshold),
        "pos_weight": float(history.pos_weight),
        "loss_name": history.loss_name,
    }
    tuned_improvement_record = {
        "dataset": dataset_name.upper(),
        "model": model_name.upper(),
        "version": "tuned_threshold_weighted_loss",
        "accuracy": float(tuned_metrics["accuracy"]),
        "precision": float(tuned_metrics["precision"]),
        "recall": float(tuned_metrics["recall"]),
        "f1_score": float(tuned_metrics["f1_score"]),
        "threshold": float(tuned_threshold),
        "pos_weight": float(history.pos_weight),
        "loss_name": history.loss_name,
    }

    return (
        baseline_predictions_df,
        tuned_predictions_df,
        baseline_metrics_record,
        tuned_metrics_record,
        runtime_record,
        threshold_record,
        improvement_record,
        pd.concat(
            [
                probability_distribution_df.assign(version="baseline_0.5_threshold"),
                probability_distribution_df.assign(version="tuned_threshold_weighted_loss"),
            ],
            ignore_index=True,
        ),
        tuned_improvement_record,
    )


def run_dataset_experiment(
    dataset_name: str,
    config: dict,
    models_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    all_runtime_rows: list[dict[str, object]] = []
    all_threshold_rows: list[dict[str, object]] = []
    all_probability_rows: list[pd.DataFrame] = []
    all_improvement_rows: list[dict[str, object]] = []
    model_names = get_enabled_deep_models(models_config)
    for seed in config["project"]["random_seeds"]:
        seed_config = clone_config_with_seed(config, int(seed))
        data_module = DataModule(seed_config)
        prepared_datasets = (
            data_module.prepare_skab_fold_datasets()
            if dataset_name.lower() == "skab"
            else [data_module.prepare_dataset(dataset_name)]
        )
        for prepared_dataset in prepared_datasets:
            for model_name in model_names:
                set_seed(int(seed))
                (
                    baseline_predictions_df,
                    tuned_predictions_df,
                    baseline_metrics_record,
                    tuned_metrics_record,
                    runtime_record,
                    threshold_record,
                    improvement_record,
                    probability_distribution_df,
                    tuned_improvement_record,
                ) = train_and_evaluate_model(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    prepared_dataset=prepared_dataset,
                    config=seed_config,
                    models_config=models_config,
                    seed=int(seed),
                )
                all_predictions.extend([baseline_predictions_df, tuned_predictions_df])
                all_metrics.extend([baseline_metrics_record, tuned_metrics_record])
                all_runtime_rows.append(runtime_record)
                all_threshold_rows.append(threshold_record)
                all_probability_rows.append(probability_distribution_df)
                all_improvement_rows.extend([improvement_record, tuned_improvement_record])

    return (
        pd.concat(all_predictions, ignore_index=True),
        pd.DataFrame(all_metrics),
        pd.DataFrame(all_runtime_rows),
        pd.DataFrame(all_threshold_rows),
        pd.concat(all_probability_rows, ignore_index=True),
        pd.DataFrame(all_improvement_rows),
    )


def save_probability_distribution_plot(probability_df: pd.DataFrame, thresholds_dir: Path) -> None:
    if probability_df.empty:
        return
    grouped = (
        probability_df.groupby(["dataset", "model", "true_label"], dropna=False)["predicted_probability"]
        .apply(list)
        .reset_index()
    )
    dataset_models = grouped[["dataset", "model"]].drop_duplicates().reset_index(drop=True)
    figure, axes = plt.subplots(len(dataset_models), 1, figsize=(8, max(4, len(dataset_models) * 3)), squeeze=False)
    for axis, (_, dataset_model) in zip(axes.flatten(), dataset_models.iterrows()):
        current = grouped[(grouped["dataset"] == dataset_model["dataset"]) & (grouped["model"] == dataset_model["model"])]
        for true_label, label_group in current.groupby("true_label", dropna=False):
            values = np.concatenate([np.asarray(item, dtype=float) for item in label_group["predicted_probability"]], axis=0)
            axis.hist(values, bins=20, alpha=0.55, label=f"true_label={int(true_label)}")
        axis.set_title(f"Probability Distribution - {dataset_model['dataset']} {dataset_model['model']}")
        axis.set_xlabel("Predicted probability")
        axis.set_ylabel("Count")
        axis.legend()
    figure.tight_layout()
    save_figure(figure, thresholds_dir / "probability_distribution.png")


def save_before_after_plot(improvement_df: pd.DataFrame, improvements_dir: Path) -> None:
    if improvement_df.empty:
        return
    summary_df = (
        improvement_df.groupby(["dataset", "model", "version"], dropna=False)["f1_score"]
        .mean()
        .reset_index()
    )
    summary_df["label"] = summary_df["dataset"].astype(str) + "-" + summary_df["model"].astype(str)
    pivoted = summary_df.pivot(index="label", columns="version", values="f1_score").fillna(0.0)
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(pivoted.index))
    width = 0.35
    baseline_values = pivoted.get("baseline_0.5_threshold", pd.Series(0.0, index=pivoted.index)).to_numpy(dtype=float)
    tuned_values = pivoted.get("tuned_threshold_weighted_loss", pd.Series(0.0, index=pivoted.index)).to_numpy(dtype=float)
    axis.bar(positions - (width / 2.0), baseline_values, width=width, label="baseline_0.5_threshold")
    axis.bar(positions + (width / 2.0), tuned_values, width=width, label="tuned_threshold_weighted_loss")
    axis.set_xticks(positions)
    axis.set_xticklabels(pivoted.index.tolist(), rotation=25, ha="right")
    axis.set_ylabel("F1 Score")
    axis.set_title("Deep Learning Before/After F1 Comparison")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, improvements_dir / "deep_learning_before_after.png")


def save_outputs(
    explanations_dir: Path,
    tables_dir: Path,
    thresholds_dir: Path,
    improvements_dir: Path,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    runtime_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    probability_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    wilcoxon_df: pd.DataFrame | None,
    mcnemar_df: pd.DataFrame | None,
    cross_family_summary_df: pd.DataFrame | None = None,
    cross_family_wilcoxon_df: pd.DataFrame | None = None,
    cross_family_mcnemar_df: pd.DataFrame | None = None,
) -> None:
    metrics_df.to_csv(tables_dir / "deep_learning_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "deep_learning_metrics_summary.csv", index=False)
    runtime_df.to_csv(tables_dir / "deep_learning_runtime_metrics.csv", index=False)
    runtime_summary_df.to_csv(tables_dir / "deep_learning_runtime_summary.csv", index=False)
    predictions_df.to_csv(explanations_dir / "deep_learning_predictions.csv", index=False)
    threshold_df.to_csv(thresholds_dir / "threshold_tuning_results.csv", index=False)
    probability_df.to_csv(thresholds_dir / "probability_distribution.csv", index=False)
    improvement_df.to_csv(improvements_dir / "deep_learning_before_after.csv", index=False)
    save_probability_distribution_plot(probability_df, thresholds_dir)
    save_before_after_plot(improvement_df, improvements_dir)
    if wilcoxon_df is not None:
        wilcoxon_df.to_csv(tables_dir / "deep_learning_wilcoxon.csv", index=False)
    if mcnemar_df is not None:
        mcnemar_df.to_csv(tables_dir / "deep_learning_mcnemar.csv", index=False)
    if cross_family_summary_df is not None:
        cross_family_summary_df.to_csv(tables_dir / "model_comparison_metrics_summary.csv", index=False)
    if cross_family_wilcoxon_df is not None:
        cross_family_wilcoxon_df.to_csv(tables_dir / "model_comparison_wilcoxon.csv", index=False)
    if cross_family_mcnemar_df is not None:
        cross_family_mcnemar_df.to_csv(tables_dir / "model_comparison_mcnemar.csv", index=False)
    with (explanations_dir / "deep_learning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": metrics_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
                "runtime": runtime_df.to_dict(orient="records"),
                "runtime_summary": runtime_summary_df.to_dict(orient="records"),
                "threshold_tuning": threshold_df.to_dict(orient="records"),
                "before_after": improvement_df.to_dict(orient="records"),
                "wilcoxon": [] if wilcoxon_df is None else wilcoxon_df.to_dict(orient="records"),
                "mcnemar": [] if mcnemar_df is None else mcnemar_df.to_dict(orient="records"),
                "cross_family_summary": [] if cross_family_summary_df is None else cross_family_summary_df.to_dict(orient="records"),
                "cross_family_wilcoxon": [] if cross_family_wilcoxon_df is None else cross_family_wilcoxon_df.to_dict(orient="records"),
                "cross_family_mcnemar": [] if cross_family_mcnemar_df is None else cross_family_mcnemar_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )


def build_statistical_outputs(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    models_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    evaluator = Evaluator()
    filtered_metrics_df = (
        metrics_df[metrics_df["version"] == "tuned_threshold_weighted_loss"].copy()
        if "version" in metrics_df.columns
        else metrics_df.copy()
    )
    filtered_predictions_df = (
        predictions_df[predictions_df["version"] == "tuned_threshold_weighted_loss"].copy()
        if "version" in predictions_df.columns
        else predictions_df.copy()
    )
    baseline_model = get_primary_deep_model_name(models_config)
    summary_df = aggregate_metrics_frame(
        filtered_metrics_df,
        group_columns=["dataset", "model", "split", "version"],
        metric_columns=["accuracy", "precision", "recall", "f1_score", "threshold", "pos_weight"],
    )
    metrics_result = evaluator.evaluate_metrics_frame(
        filtered_metrics_df,
        group_columns=["dataset", "split", "seed"],
        metric_columns=["accuracy", "precision", "recall", "f1_score"],
        baseline_model=baseline_model,
    )
    prediction_result = evaluator.evaluate_predictions_frame(
        filtered_predictions_df,
        group_columns=["dataset", "model", "split", "seed"],
        score_column="predicted_probability",
        model_column="model",
        comparison_group_columns=["dataset", "split", "seed"],
        match_columns=["row_index"],
        baseline_model=baseline_model,
    )
    return summary_df, metrics_result.statistical_tests, prediction_result.statistical_tests


def build_automata_prediction_frame(automata_explanations_df: pd.DataFrame) -> pd.DataFrame:
    return automata_explanations_df.loc[
        :,
        [
            "dataset",
            "model",
            "split",
            "seed",
            "row_index",
            "true_label",
            "predicted_label",
        ],
    ].assign(predicted_probability=np.nan, version="automata_reference")


def build_cross_family_statistical_outputs(
    deep_metrics_df: pd.DataFrame,
    deep_predictions_df: pd.DataFrame,
    config: dict,
    models_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    evaluator = Evaluator()
    automata_metrics_frames: list[pd.DataFrame] = []
    automata_prediction_frames: list[pd.DataFrame] = []
    filtered_deep_metrics_df = (
        deep_metrics_df[deep_metrics_df["version"] == "tuned_threshold_weighted_loss"].copy()
        if "version" in deep_metrics_df.columns
        else deep_metrics_df.copy()
    )
    filtered_deep_predictions_df = (
        deep_predictions_df[deep_predictions_df["version"] == "tuned_threshold_weighted_loss"].copy()
        if "version" in deep_predictions_df.columns
        else deep_predictions_df.copy()
    )

    for seed in config["project"]["random_seeds"]:
        seed_config = clone_config_with_seed(config, int(seed))
        skab_explanations, skab_metrics, _ = run_skab_experiment(seed_config, models_config)
        batadal_explanations, batadal_metrics, _ = run_batadal_experiment(seed_config, models_config)

        for explanations_df, metrics_df in (
            (skab_explanations, skab_metrics),
            (batadal_explanations, batadal_metrics),
        ):
            explanations_copy = explanations_df.copy()
            metrics_copy = metrics_df.copy()
            explanations_copy["seed"] = int(seed)
            metrics_copy["seed"] = int(seed)
            automata_prediction_frames.append(build_automata_prediction_frame(explanations_copy))
            automata_metrics_frames.append(metrics_copy)

    automata_metrics_df = pd.concat(
        [frame for frame in automata_metrics_frames if not frame.empty],
        ignore_index=True,
    )
    automata_predictions_df = pd.concat(
        [frame for frame in automata_prediction_frames if not frame.empty],
        ignore_index=True,
    )

    combined_metrics_df = pd.concat([filtered_deep_metrics_df.copy(), automata_metrics_df], ignore_index=True, sort=False)
    combined_predictions_df = pd.concat([filtered_deep_predictions_df.copy(), automata_predictions_df], ignore_index=True, sort=False)
    summary_df = aggregate_metrics_frame(
        combined_metrics_df,
        group_columns=["dataset", "model", "split"],
        metric_columns=["accuracy", "precision", "recall", "f1_score"],
    )
    metrics_result = evaluator.evaluate_metrics_frame(
        combined_metrics_df,
        group_columns=["dataset", "split", "seed"],
        metric_columns=["accuracy", "precision", "recall", "f1_score"],
        baseline_model=None,
    )
    prediction_result = evaluator.evaluate_predictions_frame(
        combined_predictions_df,
        group_columns=["dataset", "model", "split", "seed"],
        model_column="model",
        comparison_group_columns=["dataset", "split", "seed"],
        match_columns=["row_index"],
        baseline_model=None,
    )
    return summary_df, metrics_result.statistical_tests, prediction_result.statistical_tests


def print_summary(
    metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wilcoxon_df: pd.DataFrame | None,
    mcnemar_df: pd.DataFrame | None,
    cross_family_summary_df: pd.DataFrame | None = None,
    cross_family_wilcoxon_df: pd.DataFrame | None = None,
    cross_family_mcnemar_df: pd.DataFrame | None = None,
) -> None:
    print("=== Deep Learning Runs ===")
    print(metrics_df.to_string(index=False))
    print()
    print("=== Threshold Tuning Results ===")
    print(threshold_df.to_string(index=False))
    print()
    print("=== Deep Learning Mean/Std Summary ===")
    print(summary_df.to_string(index=False))
    if wilcoxon_df is not None and not wilcoxon_df.empty:
        print()
        print("=== Wilcoxon Tests ===")
        print(wilcoxon_df.to_string(index=False))
    if mcnemar_df is not None and not mcnemar_df.empty:
        print()
        print("=== McNemar Tests ===")
        print(mcnemar_df.to_string(index=False))
    if cross_family_summary_df is not None and not cross_family_summary_df.empty:
        print()
        print("=== Cross-Family Mean/Std Summary ===")
        print(cross_family_summary_df.to_string(index=False))
    if cross_family_wilcoxon_df is not None and not cross_family_wilcoxon_df.empty:
        print()
        print("=== Cross-Family Wilcoxon Tests ===")
        print(cross_family_wilcoxon_df.to_string(index=False))
    if cross_family_mcnemar_df is not None and not cross_family_mcnemar_df.empty:
        print()
        print("=== Cross-Family McNemar Tests ===")
        print(cross_family_mcnemar_df.to_string(index=False))


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    set_seed(get_primary_seed(config))
    explanations_dir, tables_dir, thresholds_dir, improvements_dir = ensure_output_dirs(config)

    skab_predictions, skab_metrics, skab_runtime, skab_thresholds, skab_probability, skab_improvements = run_dataset_experiment("skab", config, models_config)
    batadal_predictions, batadal_metrics, batadal_runtime, batadal_thresholds, batadal_probability, batadal_improvements = run_dataset_experiment("batadal", config, models_config)

    metrics_df = pd.concat([skab_metrics, batadal_metrics], ignore_index=True)
    predictions_df = pd.concat([skab_predictions, batadal_predictions], ignore_index=True)
    runtime_df = pd.concat([skab_runtime, batadal_runtime], ignore_index=True)
    threshold_df = pd.concat([skab_thresholds, batadal_thresholds], ignore_index=True)
    probability_df = pd.concat([skab_probability, batadal_probability], ignore_index=True)
    improvement_df = pd.concat([skab_improvements, batadal_improvements], ignore_index=True)

    runtime_summary_df = aggregate_metrics_frame(
        runtime_df,
        group_columns=["dataset", "model", "family", "split"],
        metric_columns=["training_time_seconds", "inference_time_seconds", "test_examples", "epochs_completed", "pos_weight"],
    )
    summary_df, wilcoxon_df, mcnemar_df = build_statistical_outputs(metrics_df, predictions_df, models_config)
    cross_family_summary_df, cross_family_wilcoxon_df, cross_family_mcnemar_df = build_cross_family_statistical_outputs(
        metrics_df,
        predictions_df,
        config,
        models_config,
    )
    save_outputs(
        explanations_dir=explanations_dir,
        tables_dir=tables_dir,
        thresholds_dir=thresholds_dir,
        improvements_dir=improvements_dir,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        runtime_df=runtime_df,
        runtime_summary_df=runtime_summary_df,
        summary_df=summary_df,
        threshold_df=threshold_df,
        probability_df=probability_df,
        improvement_df=improvement_df,
        wilcoxon_df=wilcoxon_df,
        mcnemar_df=mcnemar_df,
        cross_family_summary_df=cross_family_summary_df,
        cross_family_wilcoxon_df=cross_family_wilcoxon_df,
        cross_family_mcnemar_df=cross_family_mcnemar_df,
    )
    print_summary(
        metrics_df,
        threshold_df,
        summary_df,
        wilcoxon_df,
        mcnemar_df,
        cross_family_summary_df,
        cross_family_wilcoxon_df,
        cross_family_mcnemar_df,
    )


if __name__ == "__main__":
    main()
