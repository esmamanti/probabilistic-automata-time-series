from __future__ import annotations

import json
import math
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
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
from evaluation.metrics import aggregate_metrics_frame
from models.automata.automata_model import ProbabilisticAutomataModel
from models.automata.explainability import build_explanation_example_payload
from utils.config import load_config
from utils.experiment_context import attach_context_to_record, build_run_context
from utils.seed import clone_config_with_seed, get_experiment_seeds, get_primary_seed


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


def get_decision_config(models_config: dict) -> dict[str, object]:
    automata_config = models_config["automata"]
    decision_config = automata_config.get("decision", {})
    score_field = str(decision_config.get("score_field", "average_log_probability"))
    if score_field not in {"average_log_probability", "path_probability"}:
        raise ValueError(
            "automata.decision.score_field must be 'average_log_probability' or 'path_probability', "
            f"got '{score_field}'"
        )

    return {
        "score_field": score_field,
        "fallback_quantile": float(decision_config.get("fallback_quantile", 0.05)),
    }


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


def derive_pattern_end_indices(
    total_rows: int,
    paa_window_size: int,
    pattern_window_size: int,
    stride: int,
    pattern_count: int,
) -> list[int]:
    end_indices: list[int] = []
    for pattern_index in range(pattern_count):
        start = pattern_index * stride * paa_window_size
        end = min(start + (pattern_window_size * paa_window_size), total_rows)
        end_indices.append(max(0, end - 1))
    return end_indices


def extract_pattern_scores(score_result: dict[str, object], score_field: str) -> list[float]:
    explanations = score_result["explanations"]
    return [float(explanation[score_field]) for explanation in explanations]


def predict_labels_from_scores(scores: list[float], threshold: float) -> list[int]:
    return [1 if score <= threshold else 0 for score in scores]


def compute_best_threshold(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("scores must not be empty")

    ordered_unique_scores = sorted({float(score) for score in scores})
    candidate_thresholds = [ordered_unique_scores[0] - 1e-12, *ordered_unique_scores]

    best_threshold = candidate_thresholds[0]
    best_f1 = -1.0
    best_recall = -1.0

    for threshold in candidate_thresholds:
        predictions = predict_labels_from_scores(scores, threshold)
        current_f1 = f1_score(labels, predictions, zero_division=0)
        current_recall = recall_score(labels, predictions, zero_division=0)
        if current_f1 > best_f1 or (math.isclose(current_f1, best_f1) and current_recall > best_recall):
            best_threshold = float(threshold)
            best_f1 = float(current_f1)
            best_recall = float(current_recall)

    return best_threshold


def fallback_threshold_from_normal_scores(scores: list[float], labels: list[int], quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"fallback quantile must be between 0 and 1, got {quantile}")

    normal_scores = [score for score, label in zip(scores, labels) if int(label) == 0]
    reference_scores = normal_scores if normal_scores else scores
    return float(np.quantile(reference_scores, quantile))


def calibrate_threshold(
    scores: list[float],
    labels: list[int],
    fallback_quantile: float,
) -> float:
    unique_labels = sorted({int(label) for label in labels})
    if len(unique_labels) >= 2:
        return compute_best_threshold(scores, labels)
    return fallback_threshold_from_normal_scores(scores, labels, fallback_quantile)


def build_explanation_frame(
    dataset_name: str,
    split_name: str,
    score_result: dict[str, object],
    true_labels: list[int],
    row_indices: list[int],
    score_field: str,
    threshold: float,
) -> pd.DataFrame:
    explanations = score_result["explanations"]
    rows: list[dict[str, object]] = []

    for explanation, true_label, row_index in zip(explanations, true_labels, row_indices):
        decision_score = float(explanation[score_field])
        predicted_label = 1 if decision_score <= threshold else 0
        rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "row_index": int(row_index),
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
                "average_log_probability": explanation["average_log_probability"],
                "confidence_score": explanation["confidence_score"],
                "decision_reason": explanation["decision_reason"],
                "decision": "anomaly" if predicted_label == 1 else "normal",
                "decision_score_field": score_field,
                "decision_score": decision_score,
                "decision_threshold": float(threshold),
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
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    preprocessing_config: dict,
    models_config: dict,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    training_started_at = perf_counter()
    pipeline = PreprocessingPipeline(preprocessing_config)
    transformed_train = pipeline.fit_transform(train_features)
    transformed_test = pipeline.transform(test_features)

    train_series = extract_1d_series(transformed_train)
    calibration_series = extract_1d_series(pipeline.transform(calibration_features))
    test_series = extract_1d_series(transformed_test)

    model = build_automata_model(models_config)
    model.fit(train_series)
    decision_config = get_decision_config(models_config)
    automata_config = models_config["automata"]

    calibration_score_result = model.score_sequence(calibration_series)
    calibration_labels = derive_pattern_labels(
        raw_labels=calibration_target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(calibration_score_result["explanations"]),
    )
    threshold = calibrate_threshold(
        scores=extract_pattern_scores(calibration_score_result, str(decision_config["score_field"])),
        labels=calibration_labels,
        fallback_quantile=float(decision_config["fallback_quantile"]),
    )
    training_time_seconds = perf_counter() - training_started_at

    inference_started_at = perf_counter()
    score_result = model.score_sequence(test_series)
    inference_time_seconds = perf_counter() - inference_started_at

    true_labels = derive_pattern_labels(
        raw_labels=test_target,
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(score_result["explanations"]),
    )
    row_indices = derive_pattern_end_indices(
        total_rows=len(test_target),
        paa_window_size=automata_config["paa"]["window_size"],
        pattern_window_size=automata_config["sliding_window"]["size"],
        stride=automata_config["sliding_window"]["stride"],
        pattern_count=len(score_result["explanations"]),
    )

    explanations_df = build_explanation_frame(
        dataset_name=dataset_name,
        split_name=split_name,
        score_result=score_result,
        true_labels=true_labels,
        row_indices=row_indices,
        score_field=str(decision_config["score_field"]),
        threshold=threshold,
    )
    explanations_df["model"] = "AUTOMATA"
    metrics = compute_metrics(explanations_df)
    metrics.update(
        {
            "dataset": dataset_name,
            "model": "AUTOMATA",
            "split": split_name,
            "decision_score_field": str(decision_config["score_field"]),
            "decision_threshold": float(threshold),
            "path_probability": float(score_result["path_probability"]),
            "average_log_probability": float(score_result["average_log_probability"]),
            "test_examples": int(len(explanations_df)),
            "seen_examples": int((explanations_df["status"] == "seen").sum()),
            "unseen_examples": int((explanations_df["status"] == "unseen").sum()),
        }
    )
    runtime_record = {
        "dataset": dataset_name,
        "model": "AUTOMATA",
        "family": "AUTOMATA",
        "split": split_name,
        "training_time_seconds": float(training_time_seconds),
        "inference_time_seconds": float(inference_time_seconds),
        "test_examples": int(len(explanations_df)),
    }
    return explanations_df, metrics, runtime_record


def run_skab_experiment(config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    all_runtime_rows: list[dict[str, object]] = []

    for fold_index, train_idx, test_idx in generate_skab_group_folds(
        dataset=dataset,
        group_column=group_column,
        target_column=target_column,
        n_splits=fold_count,
        random_state=get_primary_seed(config),
    ):
        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        test_df = dataset.iloc[test_idx].reset_index(drop=True)
        train_features, train_target = split_features_and_target(train_df, feature_columns, target_column)
        test_features, test_target = split_features_and_target(test_df, feature_columns, target_column)

        explanations_df, metrics, runtime_record = run_single_automata_flow(
            dataset_name="SKAB",
            split_name=f"fold_{fold_index}",
            train_features=train_features,
            train_target=train_target,
            calibration_features=train_features,
            calibration_target=train_target,
            test_features=test_features,
            test_target=test_target,
            preprocessing_config=preprocessing_config,
            models_config=models_config,
        )
        metrics = attach_context_to_record(
            metrics,
            build_run_context(
                config=config,
                models_config=models_config,
                dataset_name="skab",
                split_name=f"fold_{fold_index}",
                seed=int(get_primary_seed(config)),
                family="AUTOMATA",
            ),
        )
        runtime_record["seed"] = int(get_primary_seed(config))
        all_explanations.append(explanations_df)
        all_metrics.append(metrics)
        all_runtime_rows.append(runtime_record)

    return pd.concat(all_explanations, ignore_index=True), pd.DataFrame(all_metrics), pd.DataFrame(all_runtime_rows)


def run_batadal_experiment(config: dict, models_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_config = config["datasets"]["batadal"]
    raw_data_path = config["paths"]["raw_data"]
    preprocessing_config = config["preprocessing"]
    dataset = load_batadal_dataset(dataset_config, raw_data_path)
    feature_columns = get_batadal_feature_columns(dataset, dataset_config)
    target_column = dataset_config["target_column"]
    splits = split_batadal_by_time(dataset, dataset_config["split"])

    train_features, train_target = split_features_and_target(splits["train"], feature_columns, target_column)
    validation_features, validation_target = split_features_and_target(
        splits["validation"],
        feature_columns,
        target_column,
    )
    test_features, test_target = split_features_and_target(splits["test"], feature_columns, target_column)

    explanations_df, metrics, runtime_record = run_single_automata_flow(
        dataset_name="BATADAL",
        split_name="test",
        train_features=train_features,
        train_target=train_target,
        calibration_features=validation_features,
        calibration_target=validation_target,
        test_features=test_features,
        test_target=test_target,
        preprocessing_config=preprocessing_config,
        models_config=models_config,
    )
    metrics = attach_context_to_record(
        metrics,
        build_run_context(
            config=config,
            models_config=models_config,
            dataset_name="batadal",
            split_name="test",
            seed=int(get_primary_seed(config)),
            family="AUTOMATA",
        ),
    )
    runtime_record["seed"] = int(get_primary_seed(config))
    return explanations_df, pd.DataFrame([metrics]), pd.DataFrame([runtime_record])


def build_automata_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "path_probability",
        "average_log_probability",
        "test_examples",
        "seen_examples",
        "unseen_examples",
    ]
    available_metric_columns = [column for column in metric_columns if column in metrics_df.columns]
    return aggregate_metrics_frame(
        metrics_df,
        group_columns=["dataset", "model", "split", "decision_score_field"],
        metric_columns=available_metric_columns,
    )


def save_outputs(
    explanations_dir: Path,
    tables_dir: Path,
    skab_explanations: pd.DataFrame,
    skab_metrics: pd.DataFrame,
    batadal_explanations: pd.DataFrame,
    batadal_metrics: pd.DataFrame,
    runtime_df: pd.DataFrame,
    runtime_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    skab_explanations.to_csv(explanations_dir / "automata_skab_explanations.csv", index=False)
    batadal_explanations.to_csv(explanations_dir / "automata_batadal_explanations.csv", index=False)
    skab_metrics.to_csv(tables_dir / "automata_skab_metrics.csv", index=False)
    batadal_metrics.to_csv(tables_dir / "automata_batadal_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "automata_metrics_summary.csv", index=False)
    runtime_df.to_csv(tables_dir / "automata_runtime_metrics.csv", index=False)
    runtime_summary_df.to_csv(tables_dir / "automata_runtime_summary.csv", index=False)

    with (explanations_dir / "automata_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "SKAB": skab_metrics.to_dict(orient="records"),
                "BATADAL": batadal_metrics.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
                "runtime": runtime_df.to_dict(orient="records"),
                "runtime_summary": runtime_summary_df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    example_source = skab_explanations if not skab_explanations.empty else batadal_explanations
    if not example_source.empty:
        example_payload = build_explanation_example_payload(example_source.iloc[0].to_dict())
        with (explanations_dir / "automata_explanation_example.json").open("w", encoding="utf-8") as handle:
            json.dump(example_payload, handle, indent=2)


def print_summary(skeb_metrics: pd.DataFrame, batadal_metrics: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("=== Automata Summary ===")
    print("SKAB folds:")
    print(skeb_metrics.to_string(index=False))
    print()
    print("BATADAL:")
    print(batadal_metrics.to_string(index=False))
    print()
    print("Aggregated Mean/Std:")
    print(summary_df.to_string(index=False))


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    models_config = load_config(PROJECT_ROOT / "configs" / "models.yaml")
    explanations_dir, tables_dir = ensure_output_dirs(config)
    all_skab_explanations: list[pd.DataFrame] = []
    all_skab_metrics: list[pd.DataFrame] = []
    all_skab_runtime: list[pd.DataFrame] = []
    all_batadal_explanations: list[pd.DataFrame] = []
    all_batadal_metrics: list[pd.DataFrame] = []
    all_batadal_runtime: list[pd.DataFrame] = []

    for seed in get_experiment_seeds(config):
        seed_config = clone_config_with_seed(config, seed)
        skab_explanations, skab_metrics, skab_runtime = run_skab_experiment(seed_config, models_config)
        batadal_explanations, batadal_metrics, batadal_runtime = run_batadal_experiment(seed_config, models_config)
        skab_explanations["seed"] = int(seed)
        skab_metrics["seed"] = int(seed)
        batadal_explanations["seed"] = int(seed)
        batadal_metrics["seed"] = int(seed)
        skab_runtime["seed"] = int(seed)
        batadal_runtime["seed"] = int(seed)
        all_skab_explanations.append(skab_explanations)
        all_skab_metrics.append(skab_metrics)
        all_skab_runtime.append(skab_runtime)
        all_batadal_explanations.append(batadal_explanations)
        all_batadal_metrics.append(batadal_metrics)
        all_batadal_runtime.append(batadal_runtime)

    skab_explanations = pd.concat(all_skab_explanations, ignore_index=True)
    skab_metrics = pd.concat(all_skab_metrics, ignore_index=True)
    batadal_explanations = pd.concat(all_batadal_explanations, ignore_index=True)
    batadal_metrics = pd.concat(all_batadal_metrics, ignore_index=True)
    runtime_df = pd.concat([*all_skab_runtime, *all_batadal_runtime], ignore_index=True)
    runtime_summary_df = aggregate_metrics_frame(
        runtime_df,
        group_columns=["dataset", "model", "family", "split"],
        metric_columns=["training_time_seconds", "inference_time_seconds", "test_examples"],
    )
    summary_df = build_automata_summary(pd.concat([skab_metrics, batadal_metrics], ignore_index=True))
    save_outputs(
        explanations_dir=explanations_dir,
        tables_dir=tables_dir,
        skab_explanations=skab_explanations,
        skab_metrics=skab_metrics,
        batadal_explanations=batadal_explanations,
        batadal_metrics=batadal_metrics,
        runtime_df=runtime_df,
        runtime_summary_df=runtime_summary_df,
        summary_df=summary_df,
    )
    print_summary(skab_metrics, batadal_metrics, summary_df)


if __name__ == "__main__":
    main()
