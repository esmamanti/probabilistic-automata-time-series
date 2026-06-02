from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule, PreparedDataset
from evaluation.evaluator import Evaluator
from evaluation.metrics import aggregate_metrics_frame
from models.deep_learning.gru_model import GRUModel
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.trainer import Trainer
from utils.config import load_config
from utils.seed import clone_config_with_seed, get_primary_seed, set_seed


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


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


def build_model(model_name: str, model_config: dict):
    if model_name == "lstm":
        return LSTMModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            output_size=model_config.get("output_size", 1),
        )
    if model_name == "gru":
        return GRUModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            output_size=model_config.get("output_size", 1),
        )
    raise ValueError(f"Unsupported deep learning model: {model_name}")


def build_trainer(model_name: str, model, config: dict, models_config: dict) -> Trainer:
    training_config = models_config["training"]
    model_config = models_config["deep_learning"][model_name]
    return Trainer(
        model=model,
        learning_rate=float(model_config["learning_rate"]),
        batch_size=int(training_config["batch_size"]),
        epochs=int(training_config["epochs"]),
        device=resolve_device(config),
        early_stopping_enabled=bool(training_config["early_stopping"]["enabled"]),
        early_stopping_patience=int(training_config["early_stopping"]["patience"]),
    )


def build_prediction_frame(
    dataset_name: str,
    model_name: str,
    split_name: str,
    prepared_dataset: PreparedDataset,
    probabilities,
    predictions,
    seed: int,
) -> pd.DataFrame:
    sequence_data = prepared_dataset.splits[split_name].sequences
    frame = prepared_dataset.splits[split_name].frame.iloc[sequence_data.sequence_end_indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": model_name,
            "split": split_name,
            "seed": int(seed),
            "row_index": sequence_data.sequence_end_indices.astype(int),
            "true_label": sequence_data.targets.astype(int),
            "predicted_label": predictions.astype(int),
            "predicted_probability": probabilities.astype(float),
        }
    ).join(frame, how="left")


def train_and_evaluate_model(
    dataset_name: str,
    model_name: str,
    prepared_dataset: PreparedDataset,
    config: dict,
    models_config: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model = build_model(model_name, models_config["deep_learning"][model_name])
    trainer = build_trainer(model_name, model, config, models_config)
    history = trainer.fit(
        train_data=prepared_dataset.splits["train"].sequences,
        validation_data=prepared_dataset.splits["validation"].sequences,
    )
    test_probabilities = trainer.predict_probabilities(prepared_dataset.splits["test"].sequences)
    test_predictions = trainer.predict_labels(prepared_dataset.splits["test"].sequences)
    metrics = trainer.evaluate(prepared_dataset.splits["test"].sequences)
    metrics.update(
        {
            "dataset": dataset_name.upper(),
            "model": model_name.upper(),
            "split": "test",
            "seed": int(seed),
            "epochs_completed": int(history.epochs_completed),
            "best_validation_loss": float(history.best_validation_loss),
            "test_examples": int(len(prepared_dataset.splits["test"].sequences.targets)),
        }
    )

    predictions_df = build_prediction_frame(
        dataset_name=dataset_name.upper(),
        model_name=model_name.upper(),
        split_name="test",
        prepared_dataset=prepared_dataset,
        probabilities=test_probabilities,
        predictions=test_predictions,
        seed=seed,
    )
    return predictions_df, metrics
def run_dataset_experiment(dataset_name: str, config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float]] = []
    for seed in config["project"]["random_seeds"]:
        seed_config = clone_config_with_seed(config, int(seed))
        prepared_dataset = DataModule(seed_config).prepare_dataset(dataset_name)
        for model_name in ("lstm", "gru"):
            set_seed(int(seed))
            predictions_df, metrics = train_and_evaluate_model(
                dataset_name=dataset_name,
                model_name=model_name,
                prepared_dataset=prepared_dataset,
                config=seed_config,
                models_config=models_config,
                seed=int(seed),
            )
            all_predictions.append(predictions_df)
            all_metrics.append(metrics)

    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(all_metrics)


def save_outputs(
    explanations_dir: Path,
    tables_dir: Path,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wilcoxon_df: pd.DataFrame | None,
    mcnemar_df: pd.DataFrame | None,
) -> None:
    metrics_df.to_csv(tables_dir / "deep_learning_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "deep_learning_metrics_summary.csv", index=False)
    if wilcoxon_df is not None:
        wilcoxon_df.to_csv(tables_dir / "deep_learning_wilcoxon.csv", index=False)
    if mcnemar_df is not None:
        mcnemar_df.to_csv(tables_dir / "deep_learning_mcnemar.csv", index=False)
    predictions_df.to_csv(explanations_dir / "deep_learning_predictions.csv", index=False)
    with (explanations_dir / "deep_learning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": metrics_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
                "wilcoxon": [] if wilcoxon_df is None else wilcoxon_df.to_dict(orient="records"),
                "mcnemar": [] if mcnemar_df is None else mcnemar_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )


def build_statistical_outputs(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    evaluator = Evaluator()
    summary_df = aggregate_metrics_frame(
        metrics_df,
        group_columns=["dataset", "model", "split"],
        metric_columns=["accuracy", "precision", "recall", "f1_score"],
    )
    metrics_result = evaluator.evaluate_metrics_frame(
        metrics_df,
        group_columns=["dataset", "split", "seed"],
        metric_columns=["accuracy", "precision", "recall", "f1_score"],
        baseline_model="LSTM",
    )
    prediction_result = evaluator.evaluate_predictions_frame(
        predictions_df,
        group_columns=["dataset", "model", "split", "seed"],
        score_column="predicted_probability",
        model_column="model",
        comparison_group_columns=["dataset", "split", "seed"],
        match_columns=["row_index"],
        baseline_model="LSTM",
    )
    return summary_df, metrics_result.statistical_tests, prediction_result.statistical_tests


def print_summary(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, wilcoxon_df: pd.DataFrame | None, mcnemar_df: pd.DataFrame | None) -> None:
    print("=== Deep Learning Runs ===")
    print(metrics_df.to_string(index=False))
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


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    set_seed(get_primary_seed(config))
    explanations_dir, tables_dir = ensure_output_dirs(config)

    skab_predictions, skab_metrics = run_dataset_experiment("skab", config, models_config)
    batadal_predictions, batadal_metrics = run_dataset_experiment("batadal", config, models_config)

    metrics_df = pd.concat([skab_metrics, batadal_metrics], ignore_index=True)
    predictions_df = pd.concat([skab_predictions, batadal_predictions], ignore_index=True)
    summary_df, wilcoxon_df, mcnemar_df = build_statistical_outputs(metrics_df, predictions_df)
    save_outputs(
        explanations_dir=explanations_dir,
        tables_dir=tables_dir,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        summary_df=summary_df,
        wilcoxon_df=wilcoxon_df,
        mcnemar_df=mcnemar_df,
    )
    print_summary(metrics_df, summary_df, wilcoxon_df, mcnemar_df)


if __name__ == "__main__":
    main()
