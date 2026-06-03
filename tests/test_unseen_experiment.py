import math

import pandas as pd

from experiments.run_unseen_experiment import compute_subset_metrics, derive_pattern_end_indices, summarize_unseen_subset


def test_derive_pattern_end_indices_matches_pattern_windows():
    end_indices = derive_pattern_end_indices(
        total_rows=10,
        paa_window_size=2,
        pattern_window_size=3,
        stride=1,
        pattern_count=3,
    )

    assert end_indices == [5, 7, 9]


def test_compute_subset_metrics_handles_empty_predictions():
    metrics = compute_subset_metrics(pd.DataFrame(columns=["true_label", "predicted_label"]))

    assert math.isnan(metrics["accuracy"])
    assert math.isnan(metrics["f1_score"])


def test_summarize_unseen_subset_reports_counts_and_scores():
    predictions_df = pd.DataFrame(
        [
            {"true_label": 1, "predicted_label": 1, "distance": 1, "confidence_score": 0.25},
            {"true_label": 0, "predicted_label": 1, "distance": 2, "confidence_score": 0.10},
        ]
    )

    summary = summarize_unseen_subset(
        predictions_df,
        dataset_name="skab",
        seed=42,
        split_name="fold_0",
        model_name="AUTOMATA",
        family="AUTOMATA",
    )

    assert summary["dataset"] == "SKAB"
    assert summary["model"] == "AUTOMATA"
    assert summary["unseen_examples"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["avg_unseen_distance"] == 1.5
    assert summary["avg_unseen_confidence"] == 0.175
