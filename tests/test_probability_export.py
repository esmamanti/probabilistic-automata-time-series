from pathlib import Path

import pandas as pd

from experiments.run_deep_models import build_probability_distribution_frame


def test_probability_distribution_csv_can_be_produced(tmp_path: Path):
    frame = build_probability_distribution_frame(
        dataset_name="SKAB",
        model_name="LSTM",
        split_name="fold_0",
        seed=42,
        probabilities=[0.1, 0.8, 0.4],
        true_labels=[0, 1, 0],
    )
    output_path = tmp_path / "probability_distribution.csv"
    frame.to_csv(output_path, index=False)

    reloaded = pd.read_csv(output_path)

    assert output_path.exists()
    assert {"dataset", "model", "split", "true_label", "predicted_probability"}.issubset(reloaded.columns)


def test_probability_values_stay_between_zero_and_one():
    frame = build_probability_distribution_frame(
        dataset_name="BATADAL",
        model_name="GRU",
        split_name="test",
        seed=7,
        probabilities=[0.0, 0.25, 0.99, 1.0],
        true_labels=[0, 0, 1, 1],
    )

    assert frame["predicted_probability"].between(0.0, 1.0).all()
