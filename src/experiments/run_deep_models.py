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
from models.deep_learning.gru_model import GRUModel
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed


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
) -> pd.DataFrame:
    sequence_data = prepared_dataset.splits[split_name].sequences
    frame = prepared_dataset.splits[split_name].frame.iloc[sequence_data.sequence_end_indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": model_name,
            "split": split_name,
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
    )
    return predictions_df, metrics


def run_dataset_experiment(dataset_name: str, config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_dataset = DataModule(config).prepare_dataset(dataset_name)
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float]] = []

    for model_name in ("lstm", "gru"):
        set_seed(config["project"]["random_seeds"][0])
        predictions_df, metrics = train_and_evaluate_model(
            dataset_name=dataset_name,
            model_name=model_name,
            prepared_dataset=prepared_dataset,
            config=config,
            models_config=models_config,
        )
        all_predictions.append(predictions_df)
        all_metrics.append(metrics)

    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(all_metrics)


def save_outputs(
    explanations_dir: Path,
    tables_dir: Path,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    metrics_df.to_csv(tables_dir / "deep_learning_metrics.csv", index=False)
    predictions_df.to_csv(explanations_dir / "deep_learning_predictions.csv", index=False)
    with (explanations_dir / "deep_learning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_df.to_dict(orient="records"), handle, indent=2)


def print_summary(metrics_df: pd.DataFrame) -> None:
    print("=== Deep Learning Summary ===")
    print(metrics_df.to_string(index=False))


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    set_seed(config["project"]["random_seeds"][0])
    explanations_dir, tables_dir = ensure_output_dirs(config)

    skab_predictions, skab_metrics = run_dataset_experiment("skab", config, models_config)
    batadal_predictions, batadal_metrics = run_dataset_experiment("batadal", config, models_config)

    metrics_df = pd.concat([skab_metrics, batadal_metrics], ignore_index=True)
    predictions_df = pd.concat([skab_predictions, batadal_predictions], ignore_index=True)
    save_outputs(explanations_dir, tables_dir, metrics_df, predictions_df)
    print_summary(metrics_df)


if __name__ == "__main__":
    main()
