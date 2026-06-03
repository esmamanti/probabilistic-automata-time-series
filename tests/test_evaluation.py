from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    aggregate_metrics_frame,
    apply_threshold,
    build_classification_report_frame,
    build_curve_frame,
    compute_classification_metrics,
    compute_confusion_metrics,
    compute_probability_metrics,
)
from evaluation.plots import (
    plot_automata_state_diagram,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_parameter_sensitivity,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_transition_probability_heatmap,
    save_figure,
)
from evaluation.statistical_tests import (
    pairwise_mcnemar_by_group,
    pairwise_wilcoxon_by_group,
    run_mcnemar_test,
    run_wilcoxon_signed_rank_test,
)
from experiments.run_automata import build_automata_summary
from experiments.run_noise_experiment import build_noise_summary
from experiments.run_parameter_analysis import build_parameter_analysis_summary
from experiments.run_deep_models import build_automata_prediction_frame, build_cross_family_statistical_outputs


def test_compute_metrics_and_confusion_outputs_expected_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.7, 0.8, 0.9]

    metrics = compute_classification_metrics(y_true, y_pred)
    confusion = compute_confusion_metrics(y_true, y_pred)
    probability_metrics = compute_probability_metrics(y_true, y_score)

    assert metrics == {"accuracy": 0.75, "precision": 2 / 3, "recall": 1.0, "f1_score": 0.8}
    assert confusion == {"true_negative": 1, "false_positive": 1, "false_negative": 0, "true_positive": 2}
    assert probability_metrics["roc_auc"] == 1.0
    assert probability_metrics["average_precision"] == 1.0


def test_report_and_curve_frames_are_built():
    report_df = build_classification_report_frame(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_score=[0.1, 0.8, 0.4, 0.2],
    )
    roc_df = build_curve_frame("roc", [0, 1, 1, 0], [0.1, 0.8, 0.4, 0.2])
    pr_df = build_curve_frame("precision_recall", [0, 1, 1, 0], [0.1, 0.8, 0.4, 0.2])

    assert {"metric", "value"} == set(report_df.columns)
    assert "false_positive_rate" in roc_df.columns
    assert "precision" in pr_df.columns
    assert apply_threshold([0.2, 0.8, 0.49], threshold=0.5).tolist() == [0, 1, 0]


def test_aggregate_metrics_frame_computes_group_statistics():
    metrics_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "A", "accuracy": 0.7, "f1_score": 0.5},
            {"dataset": "SKAB", "model": "A", "accuracy": 0.9, "f1_score": 0.7},
        ]
    )

    aggregated = aggregate_metrics_frame(metrics_df, group_columns=["dataset", "model"], metric_columns=["accuracy", "f1_score"])

    assert aggregated.loc[0, "accuracy_mean"] == 0.8
    assert aggregated.loc[0, "f1_score_max"] == 0.7


def test_experiment_summary_builders_compute_mean_and_std_outputs():
    automata_metrics = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "AUTOMATA", "split": "fold_0", "decision_score_field": "path_probability", "accuracy": 0.8, "f1_score": 0.7, "unseen_examples": 2},
            {"dataset": "SKAB", "model": "AUTOMATA", "split": "fold_0", "decision_score_field": "path_probability", "accuracy": 0.6, "f1_score": 0.5, "unseen_examples": 4},
        ]
    )
    noise_metrics = pd.DataFrame(
        [
            {"dataset": "SKAB", "family": "DEEP", "model": "LSTM", "split": "fold_0", "scenario": "noise", "accuracy": 0.8, "f1_score": 0.7},
            {"dataset": "SKAB", "family": "DEEP", "model": "LSTM", "split": "fold_0", "scenario": "noise", "accuracy": 0.6, "f1_score": 0.5},
        ]
    )
    parameter_metrics = pd.DataFrame(
        [
            {"dataset": "SKAB", "split": "fold_0", "window_size": 4, "alphabet_size": 3, "accuracy": 0.8, "state_count": 10},
            {"dataset": "SKAB", "split": "fold_0", "window_size": 4, "alphabet_size": 3, "accuracy": 0.6, "state_count": 14},
        ]
    )

    automata_summary = build_automata_summary(automata_metrics)
    noise_summary = build_noise_summary(noise_metrics)
    parameter_summary = build_parameter_analysis_summary(parameter_metrics)

    assert automata_summary.loc[0, "accuracy_mean"] == 0.7
    assert automata_summary.loc[0, "unseen_examples_mean"] == 3.0
    assert noise_summary.loc[0, "f1_score_mean"] == 0.6
    assert parameter_summary.loc[0, "state_count_mean"] == 12.0


def test_cross_family_statistical_outputs_include_automata(monkeypatch):
    deep_metrics_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "LSTM", "split": "fold_0", "seed": 42, "accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8},
            {"dataset": "SKAB", "model": "GRU", "split": "fold_0", "seed": 42, "accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": 0.6},
        ]
    )
    deep_predictions_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "LSTM", "split": "fold_0", "seed": 42, "row_index": 15, "true_label": 0, "predicted_label": 0, "predicted_probability": 0.1},
            {"dataset": "SKAB", "model": "GRU", "split": "fold_0", "seed": 42, "row_index": 15, "true_label": 0, "predicted_label": 1, "predicted_probability": 0.7},
        ]
    )
    automata_explanations = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "AUTOMATA", "split": "fold_0", "seed": 42, "row_index": 15, "true_label": 0, "predicted_label": 0},
        ]
    )
    automata_metrics = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "AUTOMATA", "split": "fold_0", "seed": 42, "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9},
        ]
    )

    monkeypatch.setattr(
        "experiments.run_deep_models.run_skab_experiment",
        lambda config, models_config: (automata_explanations.copy(), automata_metrics.copy()),
    )
    monkeypatch.setattr(
        "experiments.run_deep_models.run_batadal_experiment",
        lambda config, models_config: (pd.DataFrame(columns=automata_explanations.columns), pd.DataFrame(columns=automata_metrics.columns)),
    )

    summary_df, wilcoxon_df, mcnemar_df = build_cross_family_statistical_outputs(
        deep_metrics_df=deep_metrics_df,
        deep_predictions_df=deep_predictions_df,
        config={"project": {"random_seeds": [42]}},
        models_config={},
    )

    assert "AUTOMATA" in summary_df["model"].values
    assert wilcoxon_df is not None
    assert mcnemar_df is not None
    assert {"LSTM", "GRU", "AUTOMATA"}.issubset(set(pd.concat([mcnemar_df["model_a"], mcnemar_df["model_b"]]).unique()))


def test_build_automata_prediction_frame_adds_probability_column():
    automata_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "model": "AUTOMATA", "split": "fold_0", "seed": 42, "row_index": 12, "true_label": 0, "predicted_label": 1}
        ]
    )

    prediction_df = build_automata_prediction_frame(automata_df)

    assert "predicted_probability" in prediction_df.columns


def test_evaluator_builds_metrics_and_statistical_results():
    predictions_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "seed": 42, "row_index": 0, "model": "LSTM", "true_label": 0, "predicted_label": 0, "predicted_probability": 0.1},
            {"dataset": "SKAB", "seed": 42, "row_index": 1, "model": "LSTM", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.9},
            {"dataset": "SKAB", "seed": 42, "row_index": 0, "model": "GRU", "true_label": 0, "predicted_label": 1, "predicted_probability": 0.8},
            {"dataset": "SKAB", "seed": 42, "row_index": 1, "model": "GRU", "true_label": 1, "predicted_label": 1, "predicted_probability": 0.7},
        ]
    )
    metrics_df = pd.DataFrame(
        [
            {"dataset": "SKAB", "split": "fold_0", "model": "A", "accuracy": 0.8, "f1_score": 0.7},
            {"dataset": "SKAB", "split": "fold_0", "model": "B", "accuracy": 0.7, "f1_score": 0.6},
            {"dataset": "SKAB", "split": "fold_1", "model": "A", "accuracy": 0.9, "f1_score": 0.8},
            {"dataset": "SKAB", "split": "fold_1", "model": "B", "accuracy": 0.6, "f1_score": 0.5},
        ]
    )
    evaluator = Evaluator()

    prediction_result = evaluator.evaluate_predictions_frame(
        predictions_df,
        group_columns=["dataset", "model", "seed"],
        score_column="predicted_probability",
        model_column="model",
        comparison_group_columns=["dataset", "seed"],
        match_columns=["row_index"],
        baseline_model="LSTM",
    )
    metric_result = evaluator.evaluate_metrics_frame(
        metrics_df,
        group_columns=["dataset"],
        metric_columns=["accuracy", "f1_score"],
        baseline_model="A",
    )

    assert len(prediction_result.metrics) == 2
    assert "roc_auc" in prediction_result.metrics.columns
    assert prediction_result.statistical_tests is not None
    assert metric_result.statistical_tests is not None
    assert set(metric_result.statistical_tests["model_a"]) == {"A"}


def test_statistical_tests_and_plots_smoke():
    test_result = run_wilcoxon_signed_rank_test([0.8, 0.9, 0.7], [0.7, 0.8, 0.6])
    mcnemar_result = run_mcnemar_test([0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 1])
    pairwise_df = pairwise_wilcoxon_by_group(
        pd.DataFrame(
            [
                {"dataset": "SKAB", "split": "f0", "model": "A", "accuracy": 0.8},
                {"dataset": "SKAB", "split": "f0", "model": "B", "accuracy": 0.7},
                {"dataset": "SKAB", "split": "f1", "model": "A", "accuracy": 0.9},
                {"dataset": "SKAB", "split": "f1", "model": "B", "accuracy": 0.6},
            ]
        ),
        metric_columns=["accuracy"],
        model_column="model",
        group_columns=["dataset", "split"],
        baseline_model="A",
    )
    mcnemar_df = pairwise_mcnemar_by_group(
        pd.DataFrame(
            [
                {"dataset": "SKAB", "seed": 42, "row_index": 0, "model": "A", "true_label": 0, "predicted_label": 0},
                {"dataset": "SKAB", "seed": 42, "row_index": 1, "model": "A", "true_label": 1, "predicted_label": 0},
                {"dataset": "SKAB", "seed": 42, "row_index": 0, "model": "B", "true_label": 0, "predicted_label": 1},
                {"dataset": "SKAB", "seed": 42, "row_index": 1, "model": "B", "true_label": 1, "predicted_label": 1},
            ]
        ),
        group_columns=["dataset", "seed"],
        match_columns=["row_index"],
        model_column="model",
        baseline_model="A",
    )

    confusion_figure = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
    roc_figure = plot_roc_curve([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.2])
    pr_figure = plot_precision_recall_curve([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.2])
    bar_figure = plot_metric_bars(pd.DataFrame([{"model": "A", "accuracy": 0.8}, {"model": "B", "accuracy": 0.7}]), x="model", y="accuracy")
    line_figure = plot_parameter_sensitivity(
        pd.DataFrame(
            [
                {"window_size": 3, "f1_score": 0.4, "dataset": "SKAB"},
                {"window_size": 4, "f1_score": 0.5, "dataset": "SKAB"},
            ]
        ),
        x="window_size",
        y="f1_score",
        hue="dataset",
    )
    state_figure = plot_automata_state_diagram(
        {0: {1: 0.7, 0: 0.3}, 1: {1: 1.0}},
        state_labels={0: "aaaa", 1: "aaab"},
        probability_threshold=0.0,
    )
    heatmap_figure = plot_transition_probability_heatmap(
        {0: {1: 0.7, 0: 0.3}, 1: {1: 1.0}},
        state_labels={0: "aaaa", 1: "aaab"},
    )
    output_dir = Path("C:/Users/nalan/OneDrive/Desktop/probabilistic-automata-time-series/results/figures/test_outputs")
    saved_path = save_figure(confusion_figure, output_dir / "confusion.png")

    assert test_result["n_pairs"] == 3
    assert mcnemar_result["test"] == "mcnemar"
    assert not pairwise_df.empty
    assert not mcnemar_df.empty
    assert saved_path.exists()
    assert roc_figure.axes
    assert pr_figure.axes
    assert bar_figure.axes
    assert line_figure.axes
    assert state_figure.axes
    assert heatmap_figure.axes
