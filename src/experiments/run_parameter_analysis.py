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
from experiments.run_automata import build_automata_model, compute_metrics, derive_pattern_labels, extract_1d_series
from utils.config import load_config
from utils.seed import clone_config_with_seed, get_experiment_seeds


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def transition_density(transition_counts: dict[int, dict[int, int]], state_count: int) -> float:
    if state_count <= 0:
        return 0.0
    actual_edges = sum(len(targets) for targets in transition_counts.values())
    possible_edges = state_count * state_count
    return float(actual_edges / possible_edges) if possible_edges else 0.0


def evaluate_parameter_setting(
    dataset_name: str,
    prepared_dataset,
    models_config: dict,
    window_size: int,
    alphabet_size: int,
    seed: int,
) -> dict[str, object]:
    modified_models = deepcopy(models_config)
    modified_models["automata"]["paa"]["window_size"] = window_size
    modified_models["automata"]["sliding_window"]["size"] = window_size
    modified_models["automata"]["sax"]["alphabet_size"] = alphabet_size

    model = build_automata_model(modified_models)
    train_series = extract_1d_series(prepared_dataset.splits["train"].features)
    test_series = extract_1d_series(prepared_dataset.splits["test"].features)
    model.fit(train_series)
    score_result = model.score_sequence(test_series)

    true_labels = derive_pattern_labels(
        raw_labels=prepared_dataset.splits["test"].target,
        paa_window_size=window_size,
        pattern_window_size=window_size,
        stride=modified_models["automata"]["sliding_window"]["stride"],
        pattern_count=len(score_result["explanations"]),
    )
    predictions_df = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": [1 if row["decision"] == "anomaly" else 0 for row in score_result["explanations"]],
        }
    )
    metrics = compute_metrics(predictions_df)
    state_count = len(model.state_generator.pattern_to_state)
    transition_counts = model.transition_counts_ or {}

    return {
        "dataset": dataset_name.upper(),
        "seed": int(seed),
        "split": prepared_dataset.evaluation_split or "test",
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        **metrics,
        "state_count": int(state_count),
        "transition_sources": int(len(transition_counts)),
        "transition_density": transition_density(transition_counts, state_count),
        "unseen_examples": int(sum(row["status"] == "unseen" for row in score_result["explanations"])),
    }


def build_parameter_analysis_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "state_count",
        "transition_sources",
        "transition_density",
        "unseen_examples",
    ]
    available_metric_columns = [column for column in metric_columns if column in results_df.columns]
    return aggregate_metrics_frame(
        results_df,
        group_columns=["dataset", "split", "window_size", "alphabet_size"],
        metric_columns=available_metric_columns,
    )


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    experiments_config = load_config(PROJECT_ROOT / "configs" / "experiments.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)

    parameter_config = experiments_config["experiments"]["parameter_analysis"]
    window_sizes = parameter_config["window_sizes"]
    alphabet_sizes = parameter_config["alphabet_sizes"]
    rows: list[dict[str, object]] = []

    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        data_module = DataModule(seed_config)
        for dataset_name in ("skab", "batadal"):
            prepared_datasets = (
                data_module.prepare_skab_fold_datasets(scenario="original")
                if dataset_name == "skab"
                else [data_module.prepare_dataset(dataset_name, scenario="original")]
            )
            for window_size in window_sizes:
                for alphabet_size in alphabet_sizes:
                    for prepared_dataset in prepared_datasets:
                        rows.append(
                            evaluate_parameter_setting(
                                dataset_name=dataset_name,
                                prepared_dataset=prepared_dataset,
                                models_config=models_config,
                                window_size=int(window_size),
                                alphabet_size=int(alphabet_size),
                                seed=int(seed),
                            )
                        )

    results_df = pd.DataFrame(rows)
    summary_df = build_parameter_analysis_summary(results_df)
    results_df.to_csv(tables_dir / "parameter_analysis_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "parameter_analysis_metrics_summary.csv", index=False)
    with (explanations_dir / "parameter_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": results_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    print("=== Parameter Analysis Summary ===")
    print(results_df.to_string(index=False))
    print()
    print("=== Parameter Analysis Aggregated Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
