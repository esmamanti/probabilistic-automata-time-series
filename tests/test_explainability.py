import pandas as pd

from experiments.run_explainability_export import build_export_frame


def test_explainability_export_keeps_probability_fields_non_null():
    explanations_df = pd.DataFrame(
        [
            {
                "dataset": "SKAB",
                "time_step": 0,
                "state": 1,
                "pattern": "abcd",
                "status": "seen",
                "mapped_to": "abcd",
                "path_probability": 0.8,
                "confidence_score": 0.8,
                "decision": "normal",
                "true_label": 0,
                "row_index": 3,
                "split": "fold_0",
                "seed": 42,
            }
        ]
    )

    export_df = build_export_frame(explanations_df)

    assert export_df["path_probability"].notna().all()
    assert export_df["confidence_score"].notna().all()
