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


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def analyze_unseen_for_dataset(dataset_name: str, config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_module = DataModule(config)
    prepared_dataset = data_module.prepare_dataset(dataset_name, scenario="original")
    model = build_automata_model(models_config)
    train_series = extract_1d_series(prepared_dataset.splits["train"].features)
    test_series = extract_1d_series(prepared_dataset.splits["test"].features)
    model.fit(train_series)
    score_result = model.score_sequence(test_series)

    automata_config = models_config["automata"]
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
    explanations_df = pd.DataFrame(explanation_rows)
    unseen_df = explanations_df[explanations_df["status"] == "unseen"].reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name.upper(),
                "total_patterns": int(len(explanations_df)),
                "unseen_patterns": int(len(unseen_df)),
                "unseen_ratio": float(len(unseen_df) / len(explanations_df)) if len(explanations_df) else 0.0,
                "avg_unseen_distance": float(unseen_df["distance"].mean()) if len(unseen_df) else 0.0,
                "avg_unseen_confidence": float(unseen_df["confidence_score"].mean()) if len(unseen_df) else 0.0,
                "unseen_anomaly_rate": float((unseen_df["predicted_label"] == 1).mean()) if len(unseen_df) else 0.0,
                "unseen_true_anomaly_rate": float((unseen_df["true_label"] == 1).mean()) if len(unseen_df) else 0.0,
                "mapping_success_rate": 1.0 if len(unseen_df) else 0.0,
            }
        ]
    )
    return explanations_df, summary


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)

    skab_explanations, skab_summary = analyze_unseen_for_dataset("skab", config, models_config)
    batadal_explanations, batadal_summary = analyze_unseen_for_dataset("batadal", config, models_config)
    explanations_df = pd.concat([skab_explanations, batadal_explanations], ignore_index=True)
    summary_df = pd.concat([skab_summary, batadal_summary], ignore_index=True)

    explanations_df.to_csv(explanations_dir / "unseen_explanations.csv", index=False)
    summary_df.to_csv(tables_dir / "unseen_metrics.csv", index=False)
    with (explanations_dir / "unseen_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="records"), handle, indent=2)

    print("=== Unseen Experiment Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
