from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evaluation.metrics import (
    DEFAULT_METRIC_COLUMNS,
    aggregate_metrics_frame,
    build_classification_report_frame,
    compute_classification_metrics,
    compute_probability_metrics,
)
from evaluation.statistical_tests import pairwise_mcnemar_by_group, pairwise_wilcoxon_by_group


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    aggregated_metrics: pd.DataFrame | None = None
    statistical_tests: pd.DataFrame | None = None


class Evaluator:
    """Evaluate prediction tables and experiment metric tables with a shared API."""

    def evaluate_predictions_frame(
        self,
        predictions_df: pd.DataFrame,
        *,
        group_columns: list[str],
        target_column: str = "true_label",
        prediction_column: str = "predicted_label",
        score_column: str | None = None,
        model_column: str | None = None,
        comparison_group_columns: list[str] | None = None,
        match_columns: list[str] | None = None,
        baseline_model: str | None = None,
    ) -> EvaluationResult:
        required_columns = [*group_columns, target_column, prediction_column]
        missing_columns = [column for column in required_columns if column not in predictions_df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns in predictions frame: {missing_columns}")

        metrics_rows: list[dict[str, object]] = []
        for group_key, group_frame in predictions_df.groupby(group_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            row = {column: value for column, value in zip(group_columns, group_key)}
            row.update(
                compute_classification_metrics(
                    y_true=group_frame[target_column],
                    y_pred=group_frame[prediction_column],
                )
            )
            row["example_count"] = int(len(group_frame))

            if score_column is not None and score_column in group_frame.columns:
                row.update(compute_probability_metrics(group_frame[target_column], group_frame[score_column]))

            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        aggregated_metrics = aggregate_metrics_frame(metrics_df, group_columns=group_columns, metric_columns=self._metric_columns(metrics_df))
        test_results = None
        if model_column is not None and model_column in predictions_df.columns and comparison_group_columns and match_columns:
            test_results = pairwise_mcnemar_by_group(
                predictions_df=predictions_df,
                group_columns=comparison_group_columns,
                match_columns=match_columns,
                model_column=model_column,
                target_column=target_column,
                prediction_column=prediction_column,
                baseline_model=baseline_model,
            )
        return EvaluationResult(
            predictions=predictions_df.copy(),
            metrics=metrics_df,
            aggregated_metrics=aggregated_metrics,
            statistical_tests=test_results,
        )

    def evaluate_metrics_frame(
        self,
        metrics_df: pd.DataFrame,
        *,
        group_columns: list[str],
        metric_columns: list[str] | None = None,
        model_column: str | None = "model",
        baseline_model: str | None = None,
    ) -> EvaluationResult:
        selected_metric_columns = metric_columns or self._metric_columns(metrics_df)
        aggregated_metrics = aggregate_metrics_frame(
            metrics_df,
            group_columns=group_columns,
            metric_columns=selected_metric_columns,
        )

        test_results = None
        if model_column is not None and model_column in metrics_df.columns and len(selected_metric_columns) > 0:
            test_results = pairwise_wilcoxon_by_group(
                metrics_df=metrics_df,
                metric_columns=selected_metric_columns,
                model_column=model_column,
                group_columns=group_columns,
                baseline_model=baseline_model,
            )

        return EvaluationResult(
            predictions=pd.DataFrame(),
            metrics=metrics_df.copy(),
            aggregated_metrics=aggregated_metrics,
            statistical_tests=test_results,
        )

    def build_report_frame(
        self,
        predictions_df: pd.DataFrame,
        *,
        target_column: str = "true_label",
        prediction_column: str = "predicted_label",
        score_column: str | None = None,
    ) -> pd.DataFrame:
        score_values = predictions_df[score_column] if score_column and score_column in predictions_df.columns else None
        return build_classification_report_frame(
            y_true=predictions_df[target_column],
            y_pred=predictions_df[prediction_column],
            y_score=score_values,
        )

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        return pd.read_csv(Path(path))

    @staticmethod
    def _metric_columns(metrics_df: pd.DataFrame) -> list[str]:
        return [column for column in DEFAULT_METRIC_COLUMNS if column in metrics_df.columns]
