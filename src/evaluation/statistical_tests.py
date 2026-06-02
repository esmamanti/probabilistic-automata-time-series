from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, wilcoxon


def run_wilcoxon_signed_rank_test(
    sample_a: Sequence[float] | np.ndarray | pd.Series,
    sample_b: Sequence[float] | np.ndarray | pd.Series,
    *,
    alternative: str = "two-sided",
) -> dict[str, float | int | str]:
    values_a = np.asarray(sample_a, dtype=float)
    values_b = np.asarray(sample_b, dtype=float)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("Wilcoxon test expects 1D samples")
    if len(values_a) != len(values_b):
        raise ValueError("Wilcoxon test expects aligned samples with the same length")
    if len(values_a) == 0:
        raise ValueError("Wilcoxon test requires at least one paired observation")

    statistic, p_value = wilcoxon(values_a, values_b, alternative=alternative, zero_method="wilcox")
    differences = values_a - values_b
    return {
        "test": "wilcoxon",
        "alternative": alternative,
        "n_pairs": int(len(values_a)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
    }


def pairwise_wilcoxon_by_group(
    metrics_df: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    model_column: str = "model",
    group_columns: Sequence[str],
    baseline_model: str | None = None,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    required_columns = [model_column, *group_columns, *metric_columns]
    missing_columns = [column for column in required_columns if column not in metrics_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in metrics frame: {missing_columns}")

    model_names = sorted(metrics_df[model_column].dropna().astype(str).unique().tolist())
    if len(model_names) < 2:
        return pd.DataFrame(
            columns=[
                "metric",
                "model_a",
                "model_b",
                "test",
                "alternative",
                "n_pairs",
                "statistic",
                "p_value",
                "mean_difference",
                "median_difference",
            ]
        )

    comparison_pairs = _resolve_model_pairs(model_names, baseline_model)
    rows: list[dict[str, object]] = []

    for metric_name in metric_columns:
        pivoted = (
            metrics_df.loc[:, [*group_columns, model_column, metric_name]]
            .pivot_table(index=list(group_columns), columns=model_column, values=metric_name, aggfunc="mean")
        )
        for model_a, model_b in comparison_pairs:
            if model_a not in pivoted.columns or model_b not in pivoted.columns:
                continue

            paired = pivoted.loc[:, [model_a, model_b]].dropna()
            if paired.empty:
                continue

            result = run_wilcoxon_signed_rank_test(
                paired[model_a].to_numpy(dtype=float),
                paired[model_b].to_numpy(dtype=float),
                alternative=alternative,
            )
            rows.append(
                {
                    "metric": metric_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def run_mcnemar_test(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    pred_a: Sequence[int] | np.ndarray | pd.Series,
    pred_b: Sequence[int] | np.ndarray | pd.Series,
    *,
    exact_threshold: int = 25,
) -> dict[str, float | int | str]:
    true_values = np.asarray(y_true, dtype=int)
    pred_a_values = np.asarray(pred_a, dtype=int)
    pred_b_values = np.asarray(pred_b, dtype=int)
    if not (true_values.ndim == pred_a_values.ndim == pred_b_values.ndim == 1):
        raise ValueError("McNemar test expects 1D inputs")
    if not (len(true_values) == len(pred_a_values) == len(pred_b_values)):
        raise ValueError("McNemar test expects aligned true/predicted arrays with the same length")
    if len(true_values) == 0:
        raise ValueError("McNemar test requires at least one observation")

    a_correct = pred_a_values == true_values
    b_correct = pred_b_values == true_values
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    discordant = b + c

    if discordant == 0:
        return {
            "test": "mcnemar",
            "variant": "degenerate",
            "n_pairs": int(len(true_values)),
            "discordant_pairs": 0,
            "b": b,
            "c": c,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    if discordant < exact_threshold:
        p_value = float(binomtest(k=min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
        statistic = float(min(b, c))
        variant = "exact"
    else:
        statistic = float(((abs(b - c) - 1.0) ** 2) / discordant)
        p_value = float(1.0 - chi2.cdf(statistic, df=1))
        variant = "chi_square_cc"

    return {
        "test": "mcnemar",
        "variant": variant,
        "n_pairs": int(len(true_values)),
        "discordant_pairs": discordant,
        "b": b,
        "c": c,
        "statistic": statistic,
        "p_value": p_value,
    }


def pairwise_mcnemar_by_group(
    predictions_df: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    match_columns: Sequence[str],
    model_column: str = "model",
    target_column: str = "true_label",
    prediction_column: str = "predicted_label",
    baseline_model: str | None = None,
) -> pd.DataFrame:
    required_columns = [*group_columns, *match_columns, model_column, target_column, prediction_column]
    missing_columns = [column for column in required_columns if column not in predictions_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in predictions frame: {missing_columns}")

    model_names = sorted(predictions_df[model_column].dropna().astype(str).unique().tolist())
    if len(model_names) < 2:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "model_a",
                "model_b",
                "test",
                "variant",
                "n_pairs",
                "discordant_pairs",
                "b",
                "c",
                "statistic",
                "p_value",
            ]
        )

    comparison_pairs = _resolve_model_pairs(model_names, baseline_model)
    rows: list[dict[str, object]] = []

    for group_key, group_frame in predictions_df.groupby(list(group_columns), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = {column: value for column, value in zip(group_columns, group_key)}

        for model_a, model_b in comparison_pairs:
            frame_a = group_frame[group_frame[model_column] == model_a].copy()
            frame_b = group_frame[group_frame[model_column] == model_b].copy()
            if frame_a.empty or frame_b.empty:
                continue

            merged = frame_a.merge(
                frame_b,
                on=list(match_columns),
                suffixes=("_a", "_b"),
                how="inner",
            )
            if merged.empty:
                continue

            if not (merged[f"{target_column}_a"].astype(int) == merged[f"{target_column}_b"].astype(int)).all():
                raise ValueError("Aligned predictions disagree on true labels; check match_columns")

            result = run_mcnemar_test(
                y_true=merged[f"{target_column}_a"].astype(int).to_numpy(),
                pred_a=merged[f"{prediction_column}_a"].astype(int).to_numpy(),
                pred_b=merged[f"{prediction_column}_b"].astype(int).to_numpy(),
            )
            rows.append(
                {
                    **group_values,
                    "model_a": model_a,
                    "model_b": model_b,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def _resolve_model_pairs(model_names: list[str], baseline_model: str | None) -> list[tuple[str, str]]:
    if baseline_model is None:
        return list(combinations(model_names, 2))
    if baseline_model not in model_names:
        raise ValueError(f"Baseline model '{baseline_model}' not found in metrics frame")
    return [(baseline_model, model_name) for model_name in model_names if model_name != baseline_model]
