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
from evaluation.plots import plot_parameter_sensitivity, save_figure
from evaluation.metrics import aggregate_metrics_frame
from experiments.run_automata import build_automata_model, compute_metrics, derive_pattern_labels, extract_1d_series
from utils.config import load_config
from utils.experiment_context import attach_context_to_record, build_run_context
from utils.seed import clone_config_with_seed, get_experiment_seeds


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def ensure_analysis_dir(config: dict) -> Path:
    analysis_dir = PROJECT_ROOT / config["paths"]["automata_analysis"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


def observed_transition_count(transition_counts: dict[int, dict[int, int]]) -> int:
    return int(sum(len(targets) for targets in transition_counts.values()))


def transition_density(transition_counts: dict[int, dict[int, int]], state_count: int) -> float:
    if state_count <= 0:
        return 0.0
    actual_edges = observed_transition_count(transition_counts)
    possible_edges = state_count * state_count
    return float(actual_edges / possible_edges) if possible_edges else 0.0


def average_transitions_per_state(transition_counts: dict[int, dict[int, int]], state_count: int) -> float:
    if state_count <= 0:
        return 0.0
    return float(observed_transition_count(transition_counts) / state_count)


def evaluate_parameter_setting(
    dataset_name: str,
    prepared_dataset,
    config: dict,
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
    transition_count = observed_transition_count(transition_counts)
    avg_transitions_per_state = average_transitions_per_state(transition_counts, state_count)

    return attach_context_to_record({
        "dataset": dataset_name.upper(),
        "seed": int(seed),
        "split": prepared_dataset.evaluation_split or "test",
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        **metrics,
        "state_count": int(state_count),
        "transition_count": int(transition_count),
        "transition_sources": int(len(transition_counts)),
        "transition_density": transition_density(transition_counts, state_count),
        "avg_transitions_per_state": avg_transitions_per_state,
        "unseen_examples": int(sum(row["status"] == "unseen" for row in score_result["explanations"])),
    }, build_run_context(
        config=config,
        models_config=models_config,
        dataset_name=dataset_name,
        split_name=prepared_dataset.evaluation_split or "test",
        seed=int(seed),
        family="AUTOMATA",
        scenario="parameter_analysis",
        extra={"window_size": int(window_size), "alphabet_size": int(alphabet_size)},
    ))


def build_parameter_analysis_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "state_count",
        "transition_count",
        "transition_sources",
        "transition_density",
        "avg_transitions_per_state",
        "unseen_examples",
    ]
    available_metric_columns = [column for column in metric_columns if column in results_df.columns]
    return aggregate_metrics_frame(
        results_df,
        group_columns=["dataset", "split", "window_size", "alphabet_size"],
        metric_columns=available_metric_columns,
    )


def save_analysis_plots(results_df: pd.DataFrame, analysis_dir: Path) -> None:
    summary_df = (
        results_df.groupby(["dataset", "window_size", "alphabet_size"], dropna=False)[
            ["state_count", "transition_density", "f1_score"]
        ]
        .mean()
        .reset_index()
    )

    state_count_figure = plot_parameter_sensitivity(
        summary_df,
        x="window_size",
        y="state_count",
        hue="alphabet_size",
        title="State Count vs Window Size",
    )
    save_figure(state_count_figure, analysis_dir / "state_count_vs_window.png")

    density_figure = plot_parameter_sensitivity(
        summary_df,
        x="window_size",
        y="transition_density",
        hue="alphabet_size",
        title="Transition Density vs Window Size",
    )
    save_figure(density_figure, analysis_dir / "transition_density_vs_window.png")

    f1_figure = plot_parameter_sensitivity(
        summary_df,
        x="window_size",
        y="f1_score",
        hue="alphabet_size",
        title="F1 vs Window Size by Alphabet Size",
    )
    save_figure(f1_figure, analysis_dir / "f1_vs_window_alphabet.png")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)
    analysis_dir = ensure_analysis_dir(config)

    parameter_config = config.get("automata_analysis", {})
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
                                config=seed_config,
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
    results_df.to_csv(analysis_dir / "state_transition_analysis.csv", index=False)
    save_analysis_plots(results_df, analysis_dir)
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
