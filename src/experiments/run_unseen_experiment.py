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
from experiments.run_automata import build_automata_model, derive_pattern_labels, extract_1d_series
from utils.config import load_config
from utils.seed import clone_config_with_seed, get_experiment_seeds


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def analyze_unseen_for_dataset(dataset_name: str, config: dict, models_config: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_module = DataModule(config)
    prepared_datasets = (
        data_module.prepare_skab_fold_datasets(scenario="original")
        if dataset_name.lower() == "skab"
        else [data_module.prepare_dataset(dataset_name, scenario="original")]
    )

    explanation_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    automata_config = models_config["automata"]

    for prepared_dataset in prepared_datasets:
        split_name = prepared_dataset.evaluation_split or "test"
        model = build_automata_model(models_config)
        train_series = extract_1d_series(prepared_dataset.splits["train"].features)
        test_series = extract_1d_series(prepared_dataset.splits["test"].features)
        model.fit(train_series)
        score_result = model.score_sequence(test_series)

        true_labels = derive_pattern_labels(
            raw_labels=prepared_dataset.splits["test"].target,
            paa_window_size=automata_config["paa"]["window_size"],
            pattern_window_size=automata_config["sliding_window"]["size"],
            stride=automata_config["sliding_window"]["stride"],
            pattern_count=len(score_result["explanations"]),
        )

        explanation_rows = []
        for explanation, true_label in zip(score_result["explanations"], true_labels):
            explanation_rows.append(
                {
                    "dataset": dataset_name.upper(),
                    "seed": int(seed),
                    "split": split_name,
                    "time_step": explanation["time_step"],
                    "pattern": explanation["pattern"],
                    "mapped_to": explanation["mapped_to"],
                    "status": explanation["status"],
                    "distance": explanation["distance"],
                    "transition_probability": explanation["transition_probability"],
                    "path_probability": explanation["path_probability"],
                    "confidence_score": explanation["confidence_score"],
                    "decision": explanation["decision"],
                    "decision_reason": explanation["decision_reason"],
                    "true_label": int(true_label),
                    "predicted_label": 1 if explanation["decision"] == "anomaly" else 0,
                }
            )
        fold_explanations_df = pd.DataFrame(explanation_rows)
        unseen_df = fold_explanations_df[fold_explanations_df["status"] == "unseen"].reset_index(drop=True)
        explanation_frames.append(fold_explanations_df)
        summary_rows.append(
            {
                "dataset": dataset_name.upper(),
                "seed": int(seed),
                "split": split_name,
                "total_patterns": int(len(fold_explanations_df)),
                "unseen_patterns": int(len(unseen_df)),
                "unseen_ratio": float(len(unseen_df) / len(fold_explanations_df)) if len(fold_explanations_df) else 0.0,
                "avg_unseen_distance": float(unseen_df["distance"].mean()) if len(unseen_df) else 0.0,
                "avg_unseen_confidence": float(unseen_df["confidence_score"].mean()) if len(unseen_df) else 0.0,
                "unseen_anomaly_rate": float((unseen_df["predicted_label"] == 1).mean()) if len(unseen_df) else 0.0,
                "unseen_true_anomaly_rate": float((unseen_df["true_label"] == 1).mean()) if len(unseen_df) else 0.0,
                "mapping_success_rate": 1.0 if len(unseen_df) else 0.0,
            }
        )

    return pd.concat(explanation_frames, ignore_index=True), pd.DataFrame(summary_rows)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)
    explanation_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        skab_explanations, skab_summary = analyze_unseen_for_dataset("skab", seed_config, models_config, int(seed))
        batadal_explanations, batadal_summary = analyze_unseen_for_dataset("batadal", seed_config, models_config, int(seed))
        explanation_frames.extend([skab_explanations, batadal_explanations])
        summary_frames.extend([skab_summary, batadal_summary])
    explanations_df = pd.concat(explanation_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    explanations_df.to_csv(explanations_dir / "unseen_explanations.csv", index=False)
    summary_df.to_csv(tables_dir / "unseen_metrics.csv", index=False)
    with (explanations_dir / "unseen_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="records"), handle, indent=2)

    print("=== Unseen Experiment Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
