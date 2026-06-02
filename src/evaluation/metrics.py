from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1_score")


def _to_numpy_1d(values: Sequence[int] | np.ndarray | pd.Series, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {array.shape}")
    return array


def validate_binary_targets(y_true: Sequence[int] | np.ndarray | pd.Series) -> np.ndarray:
    targets = _to_numpy_1d(y_true, dtype=int)
    unique_values = set(np.unique(targets).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Expected binary labels containing only 0/1, got {sorted(unique_values)}")
    return targets


def validate_aligned_binary_predictions(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    true_array = validate_binary_targets(y_true)
    pred_array = _to_numpy_1d(y_pred, dtype=int)
    if len(true_array) != len(pred_array):
        raise ValueError("y_true and y_pred must have the same length")

    unique_values = set(np.unique(pred_array).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Expected binary predictions containing only 0/1, got {sorted(unique_values)}")
    return true_array, pred_array


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
) -> dict[str, float]:
    true_array, pred_array = validate_aligned_binary_predictions(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(true_array, pred_array)),
        "precision": float(precision_score(true_array, pred_array, zero_division=0)),
        "recall": float(recall_score(true_array, pred_array, zero_division=0)),
        "f1_score": float(f1_score(true_array, pred_array, zero_division=0)),
    }


def compute_confusion_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
) -> dict[str, int]:
    true_array, pred_array = validate_aligned_binary_predictions(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(true_array, pred_array, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def compute_probability_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    true_array = validate_binary_targets(y_true)
    score_array = _to_numpy_1d(y_score, dtype=float)
    if len(true_array) != len(score_array):
        raise ValueError("y_true and y_score must have the same length")

    metrics: dict[str, float] = {}
    if len(np.unique(true_array)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(true_array, score_array))
        metrics["average_precision"] = float(average_precision_score(true_array, score_array))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def apply_threshold(
    y_score: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> np.ndarray:
    score_array = _to_numpy_1d(y_score, dtype=float)
    return (score_array >= threshold).astype(int)


def build_classification_report_frame(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    *,
    y_score: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rows.extend({"metric": name, "value": value} for name, value in compute_classification_metrics(y_true, y_pred).items())
    rows.extend({"metric": name, "value": value} for name, value in compute_confusion_metrics(y_true, y_pred).items())
    if y_score is not None:
        rows.extend({"metric": name, "value": value} for name, value in compute_probability_metrics(y_true, y_score).items())
    return pd.DataFrame(rows)


def aggregate_metrics_frame(
    metrics_df: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str] = DEFAULT_METRIC_COLUMNS,
) -> pd.DataFrame:
    missing_columns = [column for column in [*group_columns, *metric_columns] if column not in metrics_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in metrics frame: {missing_columns}")

    aggregated = (
        metrics_df.loc[:, [*group_columns, *metric_columns]]
        .groupby(list(group_columns), dropna=False)
        .agg(["mean", "std", "min", "max"])
    )
    aggregated.columns = [f"{metric}_{statistic}" for metric, statistic in aggregated.columns]
    return aggregated.reset_index()


def build_curve_frame(
    curve_name: str,
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    true_array = validate_binary_targets(y_true)
    score_array = _to_numpy_1d(y_score, dtype=float)
    if len(true_array) != len(score_array):
        raise ValueError("y_true and y_score must have the same length")
    if len(np.unique(true_array)) < 2:
        raise ValueError(f"{curve_name} requires both positive and negative classes")

    if curve_name == "roc":
        x_values, y_values, thresholds = roc_curve(true_array, score_array)
        x_name, y_name = "false_positive_rate", "true_positive_rate"
    elif curve_name == "precision_recall":
        y_values, x_values, thresholds = precision_recall_curve(true_array, score_array)
        x_name, y_name = "recall", "precision"
        thresholds = np.append(thresholds, np.nan)
    else:
        raise ValueError(f"Unsupported curve name: {curve_name}")

    return pd.DataFrame(
        {
            x_name: x_values.astype(float),
            y_name: y_values.astype(float),
            "threshold": np.asarray(thresholds, dtype=float),
            "curve": curve_name,
        }
    )
