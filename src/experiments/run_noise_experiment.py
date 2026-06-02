from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule
from experiments.run_automata import build_automata_model, compute_metrics, derive_pattern_labels, extract_1d_series
from experiments.run_deep_models import build_model, build_trainer, resolve_device
from utils.config import load_config
from utils.seed import clone_config_with_seed, get_experiment_seeds, get_primary_seed, set_seed


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def run_deep_noise_experiment_for_dataset(
    dataset_name: str,
    config: dict,
    models_config: dict,
) -> pd.DataFrame:
    data_module = DataModule(config)
    original_dataset = data_module.prepare_dataset(dataset_name, scenario="original")
    noisy_dataset = data_module.prepare_dataset(dataset_name, scenario="noise")
    metrics_rows: list[dict[str, object]] = []

    for model_name in ("lstm", "gru"):
        set_seed(get_primary_seed(config))
        model = build_model(model_name, models_config["deep_learning"][model_name])
        trainer = build_trainer(model_name, model, config, models_config)
        trainer.fit(
            train_data=original_dataset.splits["train"].sequences,
            validation_data=original_dataset.splits["validation"].sequences,
        )

        for scenario_name, prepared_dataset in (("original", original_dataset), ("noise", noisy_dataset)):
            metrics = trainer.evaluate(prepared_dataset.splits["test"].sequences)
            metrics_rows.append(
                {
                    "dataset": dataset_name.upper(),
                    "model": model_name.upper(),
                    "scenario": scenario_name,
                    "family": "DEEP",
                    **metrics,
                    "test_examples": int(len(prepared_dataset.splits["test"].sequences.targets)),
                }
            )

    return pd.DataFrame(metrics_rows)


def run_automata_noise_experiment_for_dataset(
    dataset_name: str,
    config: dict,
    models_config: dict,
) -> pd.DataFrame:
    data_module = DataModule(config)
    original_dataset = data_module.prepare_dataset(dataset_name, scenario="original")
    noisy_dataset = data_module.prepare_dataset(dataset_name, scenario="noise")
    model = build_automata_model(models_config)
    train_series = extract_1d_series(original_dataset.splits["train"].features)
    model.fit(train_series)
    automata_config = models_config["automata"]
    metrics_rows: list[dict[str, object]] = []

    for scenario_name, prepared_dataset in (("original", original_dataset), ("noise", noisy_dataset)):
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
        metrics_rows.append(
            {
                "dataset": dataset_name.upper(),
                "model": "AUTOMATA",
                "scenario": scenario_name,
                "family": "AUTOMATA",
                **metrics,
                "test_examples": int(len(explanations_df)),
                "unseen_examples": int(sum(row["status"] == "unseen" for row in score_result["explanations"])),
            }
        )

    return pd.DataFrame(metrics_rows)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)
    results: list[pd.DataFrame] = []
    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        set_seed(int(seed))
        seed_results = [
            run_deep_noise_experiment_for_dataset("skab", seed_config, models_config),
            run_deep_noise_experiment_for_dataset("batadal", seed_config, models_config),
            run_automata_noise_experiment_for_dataset("skab", seed_config, models_config),
            run_automata_noise_experiment_for_dataset("batadal", seed_config, models_config),
        ]
        for frame in seed_results:
            frame["seed"] = int(seed)
            results.append(frame)
    metrics_df = pd.concat(results, ignore_index=True)
    metrics_df.to_csv(tables_dir / "noise_experiment_metrics.csv", index=False)
    with (explanations_dir / "noise_experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_df.to_dict(orient="records"), handle, indent=2)

    print("=== Noise Experiment Summary ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
