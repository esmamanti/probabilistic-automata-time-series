import json

import pandas as pd

from experiments.run_explainability_export import build_counterfactual_payload


def test_counterfactual_payload_matches_required_schema():
    explanations_df = pd.DataFrame(
        [
            {
                "pattern": "abdc",
                "mapped_to": "abcc",
                "status": "unseen",
                "decision": "anomaly",
                "rule_based_decision": "anomaly",
                "path_probability": 0.02,
                "transition_probability": 0.4,
                "distance": 1,
            }
        ]
    )

    payload = build_counterfactual_payload(explanations_df, anomaly_threshold=0.1)

    assert len(payload) == 1
    assert set(payload[0]) == {
        "original_pattern",
        "original_decision",
        "mapped_pattern",
        "counterfactual_decision",
        "original_probability",
        "counterfactual_probability",
        "levenshtein_distance",
    }
    json.dumps(payload)
