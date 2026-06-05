from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.statistical_tests import pairwise_mcnemar_by_group, pairwise_wilcoxon_by_group
from utils.config import load_config


REQUESTED_MODEL_PAIRS = [
    ("LSTM", "GRU"),
    ("LSTM", "CNN"),
    ("LSTM", "AUTOMATA"),
    ("GRU", "CNN"),
    ("GRU", "AUTOMATA"),
    ("CNN", "AUTOMATA"),
]
MODEL_PAIR_LOOKUP = {frozenset(pair): pair for pair in REQUESTED_MODEL_PAIRS}


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    explanations_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, explanations_dir


def _normalize_model_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    normalized = results_df.copy()
    normalized["model_a"] = normalized["model_a"].astype(str).str.upper()
    normalized["model_b"] = normalized["model_b"].astype(str).str.upper()
    return normalized


def _filter_requested_pairs(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    filtered = _normalize_model_pairs(results_df)
    filtered["pair_key"] = filtered.apply(lambda row: frozenset((row["model_a"], row["model_b"])), axis=1)
    filtered = filtered[filtered["pair_key"].isin(MODEL_PAIR_LOOKUP)].copy()
    if filtered.empty:
        return filtered.drop(columns=["pair_key"])

    filtered[["model_a", "model_b"]] = filtered["pair_key"].map(MODEL_PAIR_LOOKUP).apply(pd.Series)
    filtered["pair_order"] = filtered["pair_key"].map(lambda key: REQUESTED_MODEL_PAIRS.index(MODEL_PAIR_LOOKUP[key]))
    filtered = filtered.sort_values("pair_order").drop(columns=["pair_key", "pair_order"]).reset_index(drop=True)
    return filtered


def _format_results(results_df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=["dataset", "model_a", "model_b", "statistic", "p_value", "significant"])

    formatted = results_df.loc[:, ["model_a", "model_b", "statistic", "p_value"]].copy()
    formatted.insert(0, "dataset", dataset_name.upper())
    formatted["significant"] = formatted["p_value"].astype(float) < 0.05
    return formatted


def load_summary_metrics(tables_dir: Path) -> pd.DataFrame:
    summary_path = tables_dir / "model_comparison_metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary metrics file: {summary_path}")
    return pd.read_csv(summary_path)


def load_batadal_raw_metrics(tables_dir: Path) -> pd.DataFrame:
    deep_metrics_path = tables_dir / "deep_learning_metrics.csv"
    automata_metrics_path = tables_dir / "automata_batadal_metrics.csv"
    if not deep_metrics_path.exists():
        raise FileNotFoundError(f"Missing deep learning metrics file: {deep_metrics_path}")
    if not automata_metrics_path.exists():
        raise FileNotFoundError(f"Missing automata metrics file: {automata_metrics_path}")

    deep_metrics_df = pd.read_csv(deep_metrics_path, low_memory=False)
    deep_filtered = deep_metrics_df[
        (deep_metrics_df["dataset"].astype(str).str.upper() == "BATADAL")
        & (deep_metrics_df["version"].astype(str) == "tuned_threshold_weighted_loss")
    ].copy()
    deep_filtered["model"] = deep_filtered["model"].astype(str).str.upper()

    automata_metrics_df = pd.read_csv(automata_metrics_path, low_memory=False)
    automata_filtered = automata_metrics_df[automata_metrics_df["dataset"].astype(str).str.upper() == "BATADAL"].copy()
    automata_filtered["model"] = automata_filtered["model"].astype(str).str.upper()

    return pd.concat([deep_filtered, automata_filtered], ignore_index=True, sort=False)


def load_automata_prediction_frames(explanations_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_name in ("automata_skab_explanations.csv", "automata_batadal_explanations.csv"):
        path = explanations_dir / file_name
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frames.append(
            frame.loc[:, ["dataset", "model", "split", "seed", "row_index", "true_label", "predicted_label"]].copy()
        )

    if not frames:
        return pd.DataFrame(columns=["dataset", "model", "split", "seed", "row_index", "true_label", "predicted_label"])

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["dataset"] = combined["dataset"].astype(str).str.upper()
    combined["model"] = combined["model"].astype(str).str.upper()
    combined["seed"] = combined["seed"].astype(int)
    combined["row_index"] = combined["row_index"].astype(int)
    combined["true_label"] = combined["true_label"].astype(int)
    combined["predicted_label"] = combined["predicted_label"].astype(int)
    return combined


def load_combined_predictions(explanations_dir: Path) -> pd.DataFrame:
    deep_predictions_path = explanations_dir / "deep_learning_predictions.csv"
    if not deep_predictions_path.exists():
        raise FileNotFoundError(f"Missing deep learning predictions file: {deep_predictions_path}")

    deep_predictions_df = pd.read_csv(deep_predictions_path, low_memory=False)
    deep_filtered = deep_predictions_df[deep_predictions_df["version"].astype(str) == "tuned_threshold_weighted_loss"].copy()
    deep_filtered = deep_filtered.loc[:, ["dataset", "model", "split", "seed", "row_index", "true_label", "predicted_label"]]
    deep_filtered["dataset"] = deep_filtered["dataset"].astype(str).str.upper()
    deep_filtered["model"] = deep_filtered["model"].astype(str).str.upper()
    deep_filtered["seed"] = deep_filtered["seed"].astype(int)
    deep_filtered["row_index"] = deep_filtered["row_index"].astype(int)
    deep_filtered["true_label"] = deep_filtered["true_label"].astype(int)
    deep_filtered["predicted_label"] = deep_filtered["predicted_label"].astype(int)

    automata_predictions_df = load_automata_prediction_frames(explanations_dir)
    if automata_predictions_df.empty:
        return deep_filtered

    return pd.concat([deep_filtered, automata_predictions_df], ignore_index=True, sort=False)


def build_wilcoxon_results(tables_dir: Path) -> pd.DataFrame:
    summary_df = load_summary_metrics(tables_dir)
    summary_df["dataset"] = summary_df["dataset"].astype(str).str.upper()
    summary_df["model"] = summary_df["model"].astype(str).str.upper()

    skab_summary_df = summary_df[summary_df["dataset"] == "SKAB"].copy()
    skab_results = pairwise_wilcoxon_by_group(
        skab_summary_df,
        metric_columns=["f1_score_mean"],
        model_column="model",
        group_columns=["split"],
        baseline_model=None,
    )
    skab_results = _filter_requested_pairs(skab_results)
    skab_formatted = _format_results(skab_results, dataset_name="SKAB")

    batadal_metrics_df = load_batadal_raw_metrics(tables_dir)
    batadal_results = pairwise_wilcoxon_by_group(
        batadal_metrics_df,
        metric_columns=["f1_score"],
        model_column="model",
        group_columns=["seed"],
        baseline_model=None,
    )
    batadal_results = _filter_requested_pairs(batadal_results)
    batadal_formatted = _format_results(batadal_results, dataset_name="BATADAL")

    return pd.concat([skab_formatted, batadal_formatted], ignore_index=True)


def build_mcnemar_results(explanations_dir: Path) -> pd.DataFrame:
    predictions_df = load_combined_predictions(explanations_dir)
    mcnemar_results = pairwise_mcnemar_by_group(
        predictions_df,
        group_columns=["dataset"],
        match_columns=["split", "seed", "row_index"],
        model_column="model",
        target_column="true_label",
        prediction_column="predicted_label",
        baseline_model=None,
    )
    filtered_results = _filter_requested_pairs(mcnemar_results)
    if filtered_results.empty:
        return pd.DataFrame(columns=["dataset", "model_a", "model_b", "statistic", "p_value", "significant"])

    formatted = filtered_results.loc[:, ["dataset", "model_a", "model_b", "statistic", "p_value"]].copy()
    formatted["dataset"] = formatted["dataset"].astype(str).str.upper()
    formatted["significant"] = formatted["p_value"].astype(float) < 0.05
    return formatted


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    tables_dir, explanations_dir = ensure_output_dirs(config)

    wilcoxon_df = build_wilcoxon_results(tables_dir)
    mcnemar_df = build_mcnemar_results(explanations_dir)

    wilcoxon_path = tables_dir / "wilcoxon_results.csv"
    mcnemar_path = tables_dir / "mcnemar_results.csv"
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    mcnemar_df.to_csv(mcnemar_path, index=False)

    print("=== Statistical Tests ===")
    print(f"Saved: {wilcoxon_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {mcnemar_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
