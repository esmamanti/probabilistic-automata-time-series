from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.load_batadal import get_batadal_feature_columns, load_batadal_dataset
from data.load_skab import get_skab_feature_columns, load_skab_dataset
from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from data.split import generate_skab_group_folds, split_batadal_by_time, split_features_and_target
from models.automata.automata_model import ProbabilisticAutomataModel
from utils.config import load_config


def ensure_output_dirs(config: dict) -> tuple[Path, Path]:
    explanations_dir = PROJECT_ROOT / config["paths"]["explanations"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    explanations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return explanations_dir, tables_dir


def build_automata_model(models_config: dict) -> ProbabilisticAutomataModel:
    automata_config = models_config["automata"]
    return ProbabilisticAutomataModel(
        paa_window_size=automata_config["paa"]["window_size"],
        alphabet_size=automata_config["sax"]["alphabet_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        smoothing=automata_config["probability"]["smoothing"],
        epsilon=automata_config["probability"]["epsilon"],
    )


def extract_1d_series(features: pd.DataFrame) -> list[float]:
    if features.shape[1] != 1:
        raise ValueError(
            "Automata pipeline expects exactly one feature column after preprocessing; "
            f"got {features.shape[1]}"
        )
    return features.iloc[:, 0].astype(float).tolist()


def derive_pattern_labels(
    raw_labels: pd.Series,
    paa_window_size: int,
    pattern_window_size: int,
    stride: int,
    pattern_count: int,
) -> list[int]:
    labels = raw_labels.astype(int).tolist()
    derived_labels: list[int] = []

    for pattern_index in range(pattern_count):
        start = pattern_index * stride * paa_window_size
        end = min(start + (pattern_window_size * paa_window_size), len(labels))
        window_labels = labels[start:end]
        derived_labels.append(int(max(window_labels)) if window_labels else 0)

    return derived_labels


def build_explanation_frame(
    dataset_name: str,
    split_name: str,
    score_result: dict[str, object],
    true_labels: list[int],
) -> pd.DataFrame:
    explanations = score_result["explanations"]
    rows: list[dict[str, object]] = []

    for explanation, true_label in zip(explanations, true_labels):
        predicted_label = 1 if explanation["decision"] == "anomaly" else 0
        rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "time_step": explanation["time_step"],
                "pattern": explanation["pattern"],
                "state": explanation["state"],
                "previous_state": explanation["previous_state"],
                "mapped_to": explanation["mapped_to"],
                "status": explanation["status"],
                "distance": explanation["distance"],
                "transition_probability": explanation["transition_probability"],
                "probability": explanation["probability"],
                "path_probability": explanation["path_probability"],
                "confidence_score": explanation["confidence_score"],
                "decision_reason": explanation["decision_reason"],
                "decision": explanation["decision"],
                "true_label": int(true_label),
                "predicted_label": predicted_label,
            }
        )

    return pd.DataFrame(rows)


def compute_metrics(explanations_df: pd.DataFrame) -> dict[str, float]:
    y_true = explanations_df["true_label"].astype(int)
    y_pred = explanations_df["predicted_label"].astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_single_automata_flow(
    dataset_name: str,
    split_name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    preprocessing_config: dict,
    models_config: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pipeline = PreprocessingPipeline(preprocessing_config)
    transformed_train = pipeline.fit_transform(train_features)
    transformed_test = pipeline.transform(test_features)

    train_series = extract_1d_series(transformed_train)
    test_series = extract_1d_series(transformed_test)

    model = build_automata_model(models_config)
    model.fit(train_series)
    score_result = model.score_sequence(test_series)

    automata_config = models_config["automata"]
    true_labels = derive_pattern_labels(
        raw_labels=test_target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(score_result["explanations"]),
    )

    explanations_df = build_explanation_frame(dataset_name, split_name, score_result, true_labels)
    metrics = compute_metrics(explanations_df)
    metrics.update(
        {
            "dataset": dataset_name,
            "split": split_name,
            "path_probability": float(score_result["path_probability"]),
            "average_log_probability": float(score_result["average_log_probability"]),
            "test_examples": int(len(explanations_df)),
            "seen_examples": int((explanations_df["status"] == "seen").sum()),
            "unseen_examples": int((explanations_df["status"] == "unseen").sum()),
        }
    )
    return explanations_df, metrics


def run_skab_experiment(config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_config = config["datasets"]["skab"]
    raw_data_path = config["paths"]["raw_data"]
    preprocessing_config = config["preprocessing"]
    dataset = load_skab_dataset(dataset_config, raw_data_path)
    feature_columns = get_skab_feature_columns(dataset, dataset_config)
    target_column = dataset_config["target_column"]
    group_column = dataset_config["group_column"]
    fold_count = min(5, dataset[group_column].nunique())

    all_explanations: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float]] = []

    for fold_index, train_idx, test_idx in generate_skab_group_folds(
        dataset=dataset,
        group_column=group_column,
        target_column=target_column,
        n_splits=fold_count,
        random_state=config["project"]["random_seeds"][0],
    ):
        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)
        train_features, train_target = split_features_and_target(train_df, feature_columns, target_column)
        test_features, test_target = split_features_and_target(test_df, feature_columns, target_column)

        explanations_df, metrics = run_single_automata_flow(
            dataset_name="SKAB",
            split_name=f"fold_{fold_index}",
            train_features=train_features,
            train_target=train_target,
            test_features=test_features,
            test_target=test_target,
            preprocessing_config=preprocessing_config,
            models_config=models_config,
        )
        all_explanations.append(explanations_df)
        all_metrics.append(metrics)

    return pd.concat(all_explanations, ignore_index=True), pd.DataFrame(all_metrics)


def run_batadal_experiment(config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_config = config["datasets"]["batadal"]
    raw_data_path = config["paths"]["raw_data"]
    preprocessing_config = config["preprocessing"]
    dataset = load_batadal_dataset(dataset_config, raw_data_path)
    feature_columns = get_batadal_feature_columns(dataset, dataset_config)
    target_column = dataset_config["target_column"]
    splits = split_batadal_by_time(dataset, dataset_config["split"])

    train_features, train_target = split_features_and_target(splits["train"], feature_columns, target_column)
    test_features, test_target = split_features_and_target(splits["test"], feature_columns, target_column)

    explanations_df, metrics = run_single_automata_flow(
        dataset_name="BATADAL",
        split_name="test",
        train_features=train_features,
        train_target=train_target,
        test_features=test_features,
        test_target=test_target,
        preprocessing_config=preprocessing_config,
        models_config=models_config,
    )
    return explanations_df, pd.DataFrame([metrics])


def save_outputs(
    explanations_dir: Path,
    tables_dir: Path,
    skab_explanations: pd.DataFrame,
    skab_metrics: pd.DataFrame,
    batadal_explanations: pd.DataFrame,
    batadal_metrics: pd.DataFrame,
) -> None:
    skab_explanations.to_csv(explanations_dir / "automata_skab_explanations.csv", index=False)
    batadal_explanations.to_csv(explanations_dir / "automata_batadal_explanations.csv", index=False)
    skab_metrics.to_csv(tables_dir / "automata_skab_metrics.csv", index=False)
    batadal_metrics.to_csv(tables_dir / "automata_batadal_metrics.csv", index=False)

    with (explanations_dir / "automata_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "SKAB": skab_metrics.to_dict(orient="records"),
                "BATADAL": batadal_metrics.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )


def print_summary(skeb_metrics: pd.DataFrame, batadal_metrics: pd.DataFrame) -> None:
    print("=== Automata Summary ===")
    print("SKAB folds:")
    print(skeb_metrics.to_string(index=False))
    print()
    print("BATADAL:")
    print(batadal_metrics.to_string(index=False))


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)

    skab_explanations, skab_metrics = run_skab_experiment(config, models_config)
    batadal_explanations, batadal_metrics = run_batadal_experiment(config, models_config)
    save_outputs(
        explanations_dir=explanations_dir,
        tables_dir=tables_dir,
        skab_explanations=skab_explanations,
        skab_metrics=skab_metrics,
        batadal_explanations=batadal_explanations,
        batadal_metrics=batadal_metrics,
    )
    print_summary(skab_metrics, batadal_metrics)


if __name__ == "__main__":
    main()
