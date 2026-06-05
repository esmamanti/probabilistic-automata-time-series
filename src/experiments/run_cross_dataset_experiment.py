from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule
from evaluation.metrics import aggregate_metrics_frame
from evaluation.plots import plot_metric_heatmap, save_figure
from experiments.run_automata import (
    build_automata_model,
    calibrate_threshold,
    compute_metrics,
    derive_pattern_labels,
    extract_1d_series,
    extract_pattern_scores,
    get_decision_config,
)
from experiments.run_deep_models import build_model, build_trainer, get_enabled_deep_models
from utils.config import load_config
from utils.seed import clone_config_with_seed, get_experiment_seeds, set_seed


def ensure_output_dir(config: dict) -> Path:
    output_dir = PROJECT_ROOT / config["paths"]["cross_dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_dataset(dataset_name: str, config: dict):
    return DataModule(config).prepare_dataset(dataset_name, scenario="original")


def evaluate_deep_cross_dataset(
    source_dataset_name: str,
    target_dataset_name: str,
    *,
    source_dataset,
    target_dataset,
    config: dict,
    models_config: dict,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in get_enabled_deep_models(models_config):
        set_seed(int(seed))
        model = build_model(model_name, models_config["deep_learning"][model_name])
        trainer = build_trainer(model_name, model, config, models_config)
        trainer.fit(
            train_data=source_dataset.splits["train"].sequences,
            validation_data=source_dataset.splits["validation"].sequences,
        )
        metrics = trainer.evaluate(target_dataset.splits["test"].sequences)
        rows.append(
            {
                "source_dataset": source_dataset_name.upper(),
                "target_dataset": target_dataset_name.upper(),
                "model": model_name.upper(),
                "family": "DEEP",
                "seed": int(seed),
                "test_examples": int(len(target_dataset.splits["test"].sequences.targets)),
                **metrics,
            }
        )
    return rows


def evaluate_automata_cross_dataset(
    source_dataset_name: str,
    target_dataset_name: str,
    *,
    source_dataset,
    target_dataset,
    models_config: dict,
    seed: int,
) -> dict[str, object]:
    model = build_automata_model(models_config)
    decision_config = get_decision_config(models_config)
    automata_config = models_config["automata"]

    train_series = extract_1d_series(source_dataset.splits["train"].features)
    validation_series = extract_1d_series(source_dataset.splits["validation"].features)
    target_series = extract_1d_series(target_dataset.splits["test"].features)
    model.fit(train_series)

    calibration_result = model.score_sequence(validation_series)
    calibration_labels = derive_pattern_labels(
        raw_labels=source_dataset.splits["validation"].target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(calibration_result["explanations"]),
    )
    threshold = calibrate_threshold(
        scores=extract_pattern_scores(calibration_result, str(decision_config["score_field"])),
        labels=calibration_labels,
        fallback_quantile=float(decision_config["fallback_quantile"]),
    )

    target_result = model.score_sequence(target_series)
    target_labels = derive_pattern_labels(
        raw_labels=target_dataset.splits["test"].target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(target_result["explanations"]),
    )
    predictions_df = pd.DataFrame(
        {
            "true_label": target_labels,
            "predicted_label": [1 if float(row[str(decision_config["score_field"])]) <= threshold else 0 for row in target_result["explanations"]],
        }
    )
    metrics = compute_metrics(predictions_df)
    return {
        "source_dataset": source_dataset_name.upper(),
        "target_dataset": target_dataset_name.upper(),
        "model": "AUTOMATA",
        "family": "AUTOMATA",
        "seed": int(seed),
        "test_examples": int(len(predictions_df)),
        **metrics,
    }


def build_cross_dataset_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["source_dataset", "target_dataset"], dropna=False)["f1_score"]
        .mean()
        .reset_index()
    )


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    if not bool(config.get("cross_dataset", {}).get("enabled", True)):
        print("Cross-dataset analysis disabled in config.")
        return

    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    output_dir = ensure_output_dir(config)
    rows: list[dict[str, object]] = []

    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        skab_dataset = prepare_dataset("skab", seed_config)
        batadal_dataset = prepare_dataset("batadal", seed_config)

        rows.extend(
            evaluate_deep_cross_dataset(
                "skab",
                "batadal",
                source_dataset=skab_dataset,
                target_dataset=batadal_dataset,
                config=seed_config,
                models_config=models_config,
                seed=int(seed),
            )
        )
        rows.extend(
            evaluate_deep_cross_dataset(
                "batadal",
                "skab",
                source_dataset=batadal_dataset,
                target_dataset=skab_dataset,
                config=seed_config,
                models_config=models_config,
                seed=int(seed),
            )
        )
        rows.append(
            evaluate_automata_cross_dataset(
                "skab",
                "batadal",
                source_dataset=skab_dataset,
                target_dataset=batadal_dataset,
                models_config=models_config,
                seed=int(seed),
            )
        )
        rows.append(
            evaluate_automata_cross_dataset(
                "batadal",
                "skab",
                source_dataset=batadal_dataset,
                target_dataset=skab_dataset,
                models_config=models_config,
                seed=int(seed),
            )
        )

    results_df = pd.DataFrame(rows)
    summary_df = aggregate_metrics_frame(
        results_df,
        group_columns=["source_dataset", "target_dataset", "family", "model"],
        metric_columns=["accuracy", "precision", "recall", "f1_score", "test_examples"],
    )
    matrix_df = build_cross_dataset_matrix(results_df)

    results_df.to_csv(output_dir / "cross_dataset_results.csv", index=False)
    summary_df.to_csv(output_dir / "cross_dataset_summary.csv", index=False)
    matrix_figure = plot_metric_heatmap(
        matrix_df,
        index="source_dataset",
        columns="target_dataset",
        values="f1_score",
        title="Cross-Dataset F1 Matrix",
        cmap="YlGnBu",
    )
    save_figure(matrix_figure, output_dir / "cross_dataset_matrix.png")

    print("=== Cross-Dataset Results ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
