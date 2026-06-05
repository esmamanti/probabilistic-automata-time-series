import numpy as np

from models.deep_learning.trainer import tune_decision_threshold


def test_threshold_tuning_selects_best_threshold():
    probabilities = np.asarray([0.10, 0.20, 0.35, 0.55, 0.75, 0.95], dtype=float)
    targets = np.asarray([0, 0, 1, 1, 1, 1], dtype=int)

    result = tune_decision_threshold(probabilities, targets, start=0.01, end=0.99, step=0.01)

    assert 0.01 <= result["threshold"] <= 0.99
    assert result["f1_score"] >= 0.0
    assert result["recall"] >= 0.0
    assert result["precision"] >= 0.0


def test_threshold_tuning_returns_threshold_within_requested_range():
    probabilities = np.asarray([0.05, 0.15, 0.25, 0.35], dtype=float)
    targets = np.asarray([0, 0, 1, 1], dtype=int)

    result = tune_decision_threshold(probabilities, targets, start=0.01, end=0.99, step=0.01)

    assert 0.01 <= result["threshold"] <= 0.99


def test_threshold_tuning_supports_recall_oriented_objective():
    probabilities = np.asarray([0.10, 0.20, 0.30, 0.40, 0.60], dtype=float)
    targets = np.asarray([0, 0, 1, 1, 1], dtype=int)

    result = tune_decision_threshold(probabilities, targets, metric="recall", start=0.01, end=0.99, step=0.01)

    assert result["metric"] == "recall"
    assert 0.01 <= result["threshold"] <= 0.99
    assert result["recall"] >= 0.0


def test_threshold_tuning_respects_min_threshold_floor():
    probabilities = np.asarray([0.02, 0.04, 0.08, 0.12, 0.18], dtype=float)
    targets = np.asarray([0, 0, 1, 1, 1], dtype=int)

    result = tune_decision_threshold(
        probabilities,
        targets,
        metric="f1",
        start=0.01,
        end=0.20,
        step=0.01,
        min_threshold=0.10,
    )

    assert result["threshold"] >= 0.10
    assert result["min_threshold"] == 0.10
