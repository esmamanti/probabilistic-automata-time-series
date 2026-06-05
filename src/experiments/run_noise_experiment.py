from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule
from evaluation.metrics import aggregate_metrics_frame
from evaluation.plots import plot_parameter_sensitivity, save_figure
from experiments.run_automata import build_automata_model, compute_metrics, derive_pattern_labels, extract_1d_series
from experiments.run_deep_models import build_model, build_trainer, get_enabled_deep_models
from utils.config import load_config
from utils.experiment_context import attach_context_to_record, build_run_context
from utils.seed import clone_config_with_seed, get_experiment_seeds, get_primary_seed, set_seed


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def ensure_noise_dir(config: dict) -> Path:
    noise_dir = PROJECT_ROOT / config["paths"]["noise_results"]
    noise_dir.mkdir(parents=True, exist_ok=True)
    return noise_dir


def build_noise_config(config: dict, noise_level: float) -> dict:
    noisy_config = deepcopy(config)
    noisy_config.setdefault("noise", {})
    noisy_config["noise"]["gaussian_std"] = float(noise_level)
    return noisy_config


def load_original_dataset(dataset_name: str, config: dict):
    data_module = DataModule(config)
    if dataset_name.lower() == "skab":
        return data_module.prepare_skab_fold_datasets(scenario="original")
    return [data_module.prepare_dataset(dataset_name, scenario="original")]


def load_noisy_dataset(dataset_name: str, config: dict, noise_level: float):
    noisy_config = build_noise_config(config, noise_level)
    data_module = DataModule(noisy_config)
    if dataset_name.lower() == "skab":
        return data_module.prepare_skab_fold_datasets(scenario="noise")
    return [data_module.prepare_dataset(dataset_name, scenario="noise")]


def run_deep_noise_experiment_for_dataset(
    dataset_name: str,
    config: dict,
    models_config: dict,
    noise_levels: list[float],
) -> pd.DataFrame:
    original_datasets = load_original_dataset(dataset_name, config)
    noisy_dataset_map = {
        float(noise_level): load_noisy_dataset(dataset_name, config, float(noise_level))
        for noise_level in noise_levels
    }
    metrics_rows: list[dict[str, object]] = []
    model_names = get_enabled_deep_models(models_config)

    for dataset_index, original_dataset in enumerate(original_datasets):
        split_name = original_dataset.evaluation_split or "test"
        for model_name in model_names:
            set_seed(get_primary_seed(config))
            model = build_model(model_name, models_config["deep_learning"][model_name])
            trainer = build_trainer(model_name, model, config, models_config)
            trainer.fit(
                train_data=original_dataset.splits["train"].sequences,
                validation_data=original_dataset.splits["validation"].sequences,
            )

            evaluation_datasets = [(0.0, "original", original_dataset)]
            for noise_level in noise_levels:
                evaluation_datasets.append((float(noise_level), "noise", noisy_dataset_map[float(noise_level)][dataset_index]))

            for noise_level, scenario_name, prepared_dataset in evaluation_datasets:
                metrics = trainer.evaluate(prepared_dataset.splits["test"].sequences)
                context = build_run_context(
                    config=config,
                    models_config=models_config,
                    dataset_name=dataset_name,
                    split_name=split_name,
                    seed=int(get_primary_seed(config)),
                    family="DEEP",
                    model_name=model_name,
                    scenario=scenario_name,
                )
                metrics_rows.append(
                    attach_context_to_record(
                        {
                            "dataset": dataset_name.upper(),
                            "model": model_name.upper(),
                            "split": split_name,
                            "scenario": scenario_name,
                            "noise_level": float(noise_level),
                            "family": "DEEP",
                            **metrics,
                            "test_examples": int(len(prepared_dataset.splits["test"].sequences.targets)),
                        },
                        context,
                    )
                )

    return pd.DataFrame(metrics_rows)


def run_automata_noise_experiment_for_dataset(
    dataset_name: str,
    config: dict,
    models_config: dict,
    noise_levels: list[float],
) -> pd.DataFrame:
    original_datasets = load_original_dataset(dataset_name, config)
    noisy_dataset_map = {
        float(noise_level): load_noisy_dataset(dataset_name, config, float(noise_level))
        for noise_level in noise_levels
    }
    metrics_rows: list[dict[str, object]] = []

    for dataset_index, original_dataset in enumerate(original_datasets):
        split_name = original_dataset.evaluation_split or "test"
        model = build_automata_model(models_config)
        train_series = extract_1d_series(original_dataset.splits["train"].features)
        model.fit(train_series)
        automata_config = models_config["automata"]

        evaluation_datasets = [(0.0, "original", original_dataset)]
        for noise_level in noise_levels:
            evaluation_datasets.append((float(noise_level), "noise", noisy_dataset_map[float(noise_level)][dataset_index]))

        for noise_level, scenario_name, prepared_dataset in evaluation_datasets:
            test_series = extract_1d_series(prepared_dataset.splits["test"].features)
            score_result = model.score_sequence(test_series)
            true_labels = derive_pattern_labels(
                raw_labels=prepared_dataset.splits["test"].target,
                paa_window_size=automata_config["paa"]["window_size"],
                pattern_window_size=automata_config["sliding_window"]["size"],
                stride=automata_config["sliding_window"]["stride"],
                pattern_count=len(score_result["explanations"]),
            )
            explanations_df = pd.DataFrame(
                {
                    "true_label": true_labels,
                    "predicted_label": [1 if row["decision"] == "anomaly" else 0 for row in score_result["explanations"]],
                }
            )
            metrics = compute_metrics(explanations_df)
            context = build_run_context(
                config=config,
                models_config=models_config,
                dataset_name=dataset_name,
                split_name=split_name,
                seed=int(get_primary_seed(config)),
                family="AUTOMATA",
                model_name=None,
                scenario=scenario_name,
            )
            metrics_rows.append(
                attach_context_to_record(
                    {
                        "dataset": dataset_name.upper(),
                        "model": "AUTOMATA",
                        "split": split_name,
                        "scenario": scenario_name,
                        "noise_level": float(noise_level),
                        "family": "AUTOMATA",
                        **metrics,
                        "test_examples": int(len(explanations_df)),
                        "unseen_examples": int(sum(row["status"] == "unseen" for row in score_result["explanations"])),
                    },
                    context,
                )
            )

    return pd.DataFrame(metrics_rows)


def build_noise_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "test_examples",
        "unseen_examples",
    ]
    available_metric_columns = [column for column in metric_columns if column in metrics_df.columns]
    group_columns = [column for column in ["dataset", "family", "model", "split", "scenario", "noise_level"] if column in metrics_df.columns]
    return aggregate_metrics_frame(
        metrics_df,
        group_columns=group_columns,
        metric_columns=available_metric_columns,
    )


def save_noise_plot(metrics_df: pd.DataFrame, noise_dir: Path) -> None:
    plotting_df = (
        metrics_df[metrics_df["scenario"] == "noise"]
        .groupby(["dataset", "model", "noise_level"], dropna=False)["f1_score"]
        .mean()
        .reset_index()
    )
    plotting_df["series_label"] = plotting_df["dataset"].astype(str) + "-" + plotting_df["model"].astype(str)
    figure = plot_parameter_sensitivity(
        plotting_df,
        x="noise_level",
        y="f1_score",
        hue="series_label",
        title="Noise Robustness by Dataset and Model",
    )
    save_figure(figure, noise_dir / "noise_robustness_plot.png")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)
    noise_dir = ensure_noise_dir(config)
    noise_levels = [float(level) for level in config.get("noise_levels", [config.get("noise", {}).get("gaussian_std", 0.05)])]
    results: list[pd.DataFrame] = []

    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        set_seed(int(seed))
        seed_results = [
            run_deep_noise_experiment_for_dataset("skab", seed_config, models_config, noise_levels),
            run_deep_noise_experiment_for_dataset("batadal", seed_config, models_config, noise_levels),
            run_automata_noise_experiment_for_dataset("skab", seed_config, models_config, noise_levels),
            run_automata_noise_experiment_for_dataset("batadal", seed_config, models_config, noise_levels),
        ]
        for frame in seed_results:
            frame["seed"] = int(seed)
            results.append(frame)

    metrics_df = pd.concat(results, ignore_index=True)
    summary_df = build_noise_summary(metrics_df)
    metrics_df.to_csv(tables_dir / "noise_experiment_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "noise_experiment_metrics_summary.csv", index=False)
    metrics_df.loc[
        :,
        [column for column in ["dataset", "model", "noise_level", "accuracy", "precision", "recall", "f1_score"] if column in metrics_df.columns],
    ].to_csv(noise_dir / "noise_robustness_results.csv", index=False)
    save_noise_plot(metrics_df, noise_dir)

    with (explanations_dir / "noise_experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": metrics_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    print("=== Noise Experiment Summary ===")
    print(metrics_df.to_string(index=False))
    print()
    print("=== Noise Aggregated Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
