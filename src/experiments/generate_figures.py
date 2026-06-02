from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.data_module import DataModule
from evaluation.plots import (
    plot_automata_state_diagram,
    plot_confusion_matrix,
    plot_parameter_sensitivity,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_transition_probability_heatmap,
    save_figure,
)
from experiments.run_automata import build_automata_model, extract_1d_series
from experiments.run_parameter_analysis import main as run_parameter_analysis_main
from utils.config import load_config
from utils.seed import get_primary_seed, set_seed


def ensure_figures_dir(config: dict) -> Path:
    figures_dir = PROJECT_ROOT / config["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def load_deep_learning_predictions() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "explanations" / "deep_learning_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Deep learning predictions file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def load_parameter_analysis_results() -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "tables" / "parameter_analysis_metrics.csv"
    if not path.exists():
        run_parameter_analysis_main()
    return pd.read_csv(path)


def select_best_prediction_slice(predictions_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["dataset", "model", "split"]
    if "seed" in predictions_df.columns:
        group_columns.append("seed")
    scored = predictions_df.assign(
        is_correct=(predictions_df["true_label"].astype(int) == predictions_df["predicted_label"].astype(int)).astype(int)
    )
    grouped = (
        scored.groupby(group_columns, dropna=False)
        .agg(accuracy=("is_correct", "mean"), count=("is_correct", "size"))
        .reset_index()
        .sort_values(["accuracy", "count"], ascending=[False, False], kind="stable")
    )
    best_row = grouped.iloc[0]
    mask = (
        (predictions_df["dataset"] == best_row["dataset"])
        & (predictions_df["model"] == best_row["model"])
        & (predictions_df["split"] == best_row["split"])
    )
    if "seed" in predictions_df.columns:
        mask = mask & (predictions_df["seed"] == best_row["seed"])
    return predictions_df[mask].reset_index(drop=True)


def build_automata_artifacts_for_skab(config: dict, models_config: dict):
    prepared_dataset = DataModule(config).prepare_dataset("skab", scenario="original")
    model = build_automata_model(models_config)
    train_series = extract_1d_series(prepared_dataset.splits["train"].features)
    model.fit(train_series)
    return model


def generate_confusion_and_curve_figures(figures_dir: Path) -> list[Path]:
    predictions_df = load_deep_learning_predictions()
    best_slice = select_best_prediction_slice(predictions_df)
    dataset_name = str(best_slice["dataset"].iloc[0])
    model_name = str(best_slice["model"].iloc[0])
    split_name = str(best_slice["split"].iloc[0])

    confusion_figure = plot_confusion_matrix(
        best_slice["true_label"],
        best_slice["predicted_label"],
        title=f"Confusion Matrix - {dataset_name} {model_name} ({split_name})",
        normalize=False,
    )
    roc_figure = plot_roc_curve(
        best_slice["true_label"],
        best_slice["predicted_probability"],
        title=f"ROC Curve - {dataset_name} {model_name} ({split_name})",
    )
    pr_figure = plot_precision_recall_curve(
        best_slice["true_label"],
        best_slice["predicted_probability"],
        title=f"Precision-Recall Curve - {dataset_name} {model_name} ({split_name})",
    )

    output_paths = [
        save_figure(confusion_figure, figures_dir / "confusion_matrix_best_model.png"),
        save_figure(roc_figure, figures_dir / "roc_curve_best_model.png"),
        save_figure(pr_figure, figures_dir / "precision_recall_curve_best_model.png"),
    ]
    return output_paths


def generate_automata_figures(figures_dir: Path, config: dict, models_config: dict) -> list[Path]:
    model = build_automata_artifacts_for_skab(config, models_config)
    transition_probabilities = model.transition_probabilities_ or {}
    state_labels = {state_id: pattern for state_id, pattern in model.state_generator.state_to_pattern.items()}

    state_diagram_figure = plot_automata_state_diagram(
        transition_probabilities,
        state_labels=state_labels,
        title="Automata State Diagram - SKAB Train",
        probability_threshold=0.05,
    )
    heatmap_figure = plot_transition_probability_heatmap(
        transition_probabilities,
        state_labels=state_labels,
        title="Transition Probability Heatmap - SKAB Train",
    )

    output_paths = [
        save_figure(state_diagram_figure, figures_dir / "automata_state_diagram_skab.png"),
        save_figure(heatmap_figure, figures_dir / "transition_probability_heatmap_skab.png"),
    ]
    return output_paths


def generate_parameter_sensitivity_figure(figures_dir: Path) -> Path:
    parameter_df = load_parameter_analysis_results()
    skab_df = parameter_df[parameter_df["dataset"] == "SKAB"].copy()
    if skab_df.empty:
        raise ValueError("Parameter analysis results do not contain SKAB rows")

    sensitivity_figure = plot_parameter_sensitivity(
        skab_df,
        x="window_size",
        y="f1_score",
        hue="alphabet_size",
        style="alphabet_size",
        title="Parameter Sensitivity - SKAB F1 by Window Size",
    )
    return save_figure(sensitivity_figure, figures_dir / "parameter_sensitivity_skab.png")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    set_seed(get_primary_seed(config))
    figures_dir = ensure_figures_dir(config)

    generated_paths: list[Path] = []
    generated_paths.extend(generate_confusion_and_curve_figures(figures_dir))
    generated_paths.extend(generate_automata_figures(figures_dir, config, models_config))
    generated_paths.append(generate_parameter_sensitivity_figure(figures_dir))

    print("=== Generated Figures ===")
    for path in generated_paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
