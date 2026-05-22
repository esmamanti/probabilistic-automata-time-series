import numpy as np
import pandas as pd

from data.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from data.preprocessing.sequence import generate_sequences


def test_preprocessing_pipeline_fits_only_on_train_data():
    pipeline = PreprocessingPipeline(
        {
            "scaler": "standard",
            "pca": {"enabled": False},
        }
    )
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    test = pd.DataFrame({"x": [10.0, 11.0], "y": [10.0, 11.0]})

    transformed_train = pipeline.fit_transform(train)
    transformed_test = pipeline.transform(test)

    assert np.allclose(transformed_train.mean().to_numpy(), np.zeros(2), atol=1e-7)
    assert transformed_test["x"].iloc[0] > 9.0


def test_sequence_generation_uses_window_max_as_label():
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([0, 0, 1, 0])

    sequences = generate_sequences(features, target, sequence_length=2, stride=1)

    assert sequences.features.shape == (3, 2, 1)
    assert sequences.targets.tolist() == [0, 1, 1]
