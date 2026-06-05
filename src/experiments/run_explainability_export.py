from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.plots import plot_histogram_by_label, save_figure
from utils.config import load_config


def ensure_output_dir(config: dict) -> Path:
    output_dir = PROJECT_ROOT / config["paths"]["explanations"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_automata_explanations(output_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_name in ("automata_skab_explanations.csv", "automata_batadal_explanations.csv"):
        path = output_dir / file_name
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError("Automata explanation CSV files were not found. Run run_automata.py first.")
    return pd.concat(frames, ignore_index=True)


def build_export_frame(explanations_df: pd.DataFrame) -> pd.DataFrame:
    export_columns = [
        "dataset",
        "time_step",
        "state",
        "pattern",
        "status",
        "mapped_to",
        "path_probability",
        "confidence_score",
        "decision",
        "true_label",
        "row_index",
        "split",
        "seed",
        "rule_based_decision",
        "decision_reason",
        "transition_probability",
        "decision_score",
        "decision_threshold",
    ]
    available_columns = [column for column in export_columns if column in explanations_df.columns]
    return explanations_df.loc[:, available_columns].copy()


def build_counterfactual_payload(explanations_df: pd.DataFrame, anomaly_threshold: float) -> list[dict[str, object]]:
    unseen_anomalies = explanations_df[
        (explanations_df["status"] == "unseen")
        & ((explanations_df["decision"] == "anomaly") | (explanations_df.get("rule_based_decision") == "anomaly"))
    ].copy()
    if unseen_anomalies.empty:
        return []

    payload: list[dict[str, object]] = []
    for _, row in unseen_anomalies.iterrows():
        counterfactual_decision = "anomaly" if float(row.get("transition_probability", 0.0)) < float(anomaly_threshold) else "normal"
        payload.append(
            {
                "original_pattern": row["pattern"],
                "original_decision": "anomaly",
                "mapped_pattern": row["mapped_to"],
                "counterfactual_decision": counterfactual_decision,
                "original_probability": float(row["path_probability"]),
                "counterfactual_probability": float(row.get("transition_probability", row["path_probability"])),
                "levenshtein_distance": int(row["distance"]),
            }
        )
    return payload


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explainability_config = config.get("explainability", {})
    anomaly_threshold = float(models_config.get("automata", {}).get("anomaly_threshold", 0.1))
    output_dir = ensure_output_dir(config)
    explanations_df = load_automata_explanations(output_dir)
    export_df = build_export_frame(explanations_df)

    if bool(explainability_config.get("save_csv", True)):
        export_df.to_csv(output_dir / "automata_explanations.csv", index=False)
    if bool(explainability_config.get("save_json", True)):
        with (output_dir / "automata_explanations.json").open("w", encoding="utf-8") as handle:
            json.dump(export_df.to_dict(orient="records"), handle, indent=2)

    histogram_df = explanations_df.copy()
    histogram_df["label_group"] = histogram_df["true_label"].map({0: "normal", 1: "anomaly"}).fillna("unknown")
    histogram_figure = plot_histogram_by_label(
        histogram_df,
        value_column="confidence_score",
        label_column="label_group",
        title="Confidence Score Distribution",
    )
    save_figure(histogram_figure, output_dir / "confidence_histogram.png")

    if bool(explainability_config.get("counterfactual", True)):
        counterfactual_payload = build_counterfactual_payload(explanations_df, anomaly_threshold)
        with (output_dir / "counterfactual_explanations.json").open("w", encoding="utf-8") as handle:
            json.dump(counterfactual_payload, handle, indent=2)

    print("=== Explainability Export Completed ===")
    print(f"Rows exported: {len(export_df)}")


if __name__ == "__main__":
    main()
