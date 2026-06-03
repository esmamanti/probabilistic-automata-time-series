from models.automata.explainability import (
    REQUIRED_EXPLANATION_FIELDS,
    build_explanation_example_payload,
    validate_explanation_record,
)
from models.automata.automata_model import ProbabilisticAutomataModel
from experiments.run_automata import calibrate_threshold, predict_labels_from_scores


def test_automata_model_produces_explanations_for_seen_patterns():
    model = ProbabilisticAutomataModel(
        paa_window_size=1,
        alphabet_size=3,
        pattern_window_size=2,
        anomaly_threshold=0.05,
        smoothing=True,
        epsilon=1e-3,
    )

    training_series = [0.0, 0.2, 0.4, 0.6, 0.8]
    model.fit(training_series)
    result = model.score_sequence(training_series)

    assert result["explanations"]
    assert result["explanations"][0]["decision"] == "normal"
    assert all("mapped_to" in explanation for explanation in result["explanations"])
    assert all("decision_reason" in explanation for explanation in result["explanations"])
    assert all("path_probability" in explanation for explanation in result["explanations"])
    assert all("average_log_probability" in explanation for explanation in result["explanations"])
    assert all(explanation["confidence_score"] == explanation["path_probability"] for explanation in result["explanations"])


def test_automata_model_marks_unseen_patterns_as_anomaly():
    model = ProbabilisticAutomataModel(
        paa_window_size=1,
        alphabet_size=4,
        pattern_window_size=2,
        anomaly_threshold=0.2,
        smoothing=True,
        epsilon=1e-3,
    )

    model.fit([0.0, 0.1, 0.2, 0.3, 0.4])
    result = model.score_sequence([0.0, 5.0, -5.0, 5.0, -5.0])

    assert any(explanation["status"] == "unseen" for explanation in result["explanations"])
    assert any(explanation["decision"] == "anomaly" for explanation in result["explanations"])
    assert any(explanation["decision_reason"] == "unseen_pattern" for explanation in result["explanations"])
    assert all(explanation["confidence_score"] == explanation["path_probability"] for explanation in result["explanations"])


def test_explanation_schema_contains_required_fields_and_json_example():
    model = ProbabilisticAutomataModel(
        paa_window_size=1,
        alphabet_size=3,
        pattern_window_size=2,
        anomaly_threshold=0.05,
        smoothing=True,
        epsilon=1e-3,
    )

    model.fit([0.0, 0.2, 0.4, 0.6, 0.8])
    result = model.score_sequence([0.0, 0.2, 0.4, 0.6, 0.8])
    explanation = validate_explanation_record(result["explanations"][0])
    example_payload = build_explanation_example_payload(explanation)

    assert set(REQUIRED_EXPLANATION_FIELDS).issubset(explanation.keys())
    assert set(
        [
            "time_step",
            "state",
            "previous_state",
            "pattern",
            "status",
            "mapped_to",
            "distance",
            "transition_probability",
            "path_probability",
            "confidence_score",
            "decision_reason",
            "decision",
        ]
    ) == set(example_payload.keys())


def test_calibrate_threshold_finds_f1_optimal_split():
    scores = [-5.0, -4.0, -3.0, -2.0]
    labels = [1, 1, 0, 0]

    threshold = calibrate_threshold(scores, labels, fallback_quantile=0.05)

    assert threshold == -4.0
    assert predict_labels_from_scores(scores, threshold) == [1, 1, 0, 0]


def test_calibrate_threshold_uses_quantile_when_only_normal_labels_exist():
    scores = [-5.0, -4.0, -3.0, -2.0]
    labels = [0, 0, 0, 0]

    threshold = calibrate_threshold(scores, labels, fallback_quantile=0.25)

    assert threshold == -4.25
