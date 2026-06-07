import numpy as np
import torch

from data.preprocessing.sequence import SequenceDataset
from experiments.run_deep_models import build_model, build_trainer, get_enabled_deep_models, resolve_threshold_tuning_config
from models.deep_learning.cnn_model import CNNModel
from models.deep_learning.gru_model import GRUModel
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.trainer import BinaryFocalLossWithLogits, Trainer


def make_sequence_dataset() -> SequenceDataset:
    features = np.asarray(
        [
            [[0.0], [0.1], [0.2]],
            [[0.1], [0.2], [0.3]],
            [[0.8], [0.9], [1.0]],
            [[0.9], [1.0], [1.1]],
        ],
        dtype=float,
    )
    targets = np.asarray([0, 0, 1, 1], dtype=int)
    end_indices = np.asarray([2, 3, 4, 5], dtype=int)
    return SequenceDataset(features=features, targets=targets, sequence_end_indices=end_indices)


def test_lstm_forward_returns_batch_logits():
    model = LSTMModel(input_size=1, hidden_size=4, num_layers=1, dropout=0.0, output_size=1)
    inputs = torch.randn(5, 3, 1)

    logits = model(inputs)

    assert logits.shape == (5,)


def test_gru_forward_returns_batch_logits():
    model = GRUModel(input_size=1, hidden_size=4, num_layers=1, dropout=0.0, output_size=1)
    inputs = torch.randn(5, 3, 1)

    logits = model(inputs)

    assert logits.shape == (5,)


def test_cnn_forward_returns_batch_logits():
    model = CNNModel(input_channels=1, num_filters=4, kernel_size=3, dropout=0.0, output_size=1)
    inputs = torch.randn(5, 3, 1)

    logits = model(inputs)

    assert logits.shape == (5,)


def test_build_model_supports_cnn_architecture_from_config():
    model = build_model(
        "cnn",
        {
            "architecture": "cnn",
            "input_channels": 1,
            "num_filters": 8,
            "kernel_size": 3,
            "dropout": 0.1,
            "output_size": 1,
        },
        input_feature_count=1,
    )

    assert isinstance(model, CNNModel)


def test_get_enabled_deep_models_reads_config_without_hardcoding():
    models_config = {
        "deep_learning": {
            "my_lstm": {"architecture": "lstm", "enabled": True},
            "my_gru": {"architecture": "gru", "enabled": False},
            "my_cnn": {"architecture": "cnn", "enabled": True},
        }
    }

    assert get_enabled_deep_models(models_config) == ["my_lstm", "my_cnn"]


def test_trainer_fits_and_evaluates_sequence_classifier():
    train_data = make_sequence_dataset()
    validation_data = make_sequence_dataset()
    test_data = make_sequence_dataset()
    model = LSTMModel(input_size=1, hidden_size=8, num_layers=1, dropout=0.0, output_size=1)
    trainer = Trainer(
        model=model,
        learning_rate=0.05,
        batch_size=2,
        epochs=5,
        device="cpu",
        early_stopping_enabled=True,
        early_stopping_patience=2,
    )

    history = trainer.fit(train_data, validation_data)
    probabilities = trainer.predict_probabilities(test_data)
    predictions = trainer.predict_labels(test_data)
    metrics = trainer.evaluate(test_data)

    assert history.epochs_completed >= 1
    assert probabilities.shape == (4,)
    assert predictions.shape == (4,)
    assert set(predictions.tolist()).issubset({0, 1})
    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score"}


def test_resolve_threshold_tuning_config_applies_dataset_floor_override():
    models_config = {
        "training": {
            "threshold_tuning": {
                "enabled": True,
                "metric": "f1",
                "beta": 2.0,
                "start": 0.01,
                "end": 0.99,
                "step": 0.01,
                "dataset_overrides": {
                    "SKAB": {"metric": "f1", "min_threshold": 0.10},
                },
            }
        }
    }

    resolved = resolve_threshold_tuning_config("skab", models_config)

    assert resolved["metric"] == "f1"
    assert resolved["min_threshold"] == 0.10


def test_build_trainer_applies_focal_loss_for_batadal_gru_override():
    model = GRUModel(input_size=1, hidden_size=4, num_layers=1, dropout=0.0, output_size=1)
    config = {"project": {"device": "cpu"}}
    models_config = {
        "training": {
            "batch_size": 2,
            "epochs": 2,
            "early_stopping": {"enabled": True, "patience": 1, "monitor": "val_f1", "mode": "max"},
            "class_imbalance": {
                "use_pos_weight": True,
                "pos_weight_strategy": "neg_pos_ratio",
                "loss_name": "bce",
                "focal_gamma": 2.0,
                "dataset_model_overrides": [
                    {"dataset": "BATADAL", "model": "GRU", "loss_name": "focal", "focal_gamma": 1.5}
                ],
            },
            "threshold_tuning": {"enabled": True},
        },
        "deep_learning": {
            "gru": {
                "architecture": "gru",
                "input_size": 1,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "output_size": 1,
                "learning_rate": 0.01,
            }
        },
    }

    trainer = build_trainer("batadal", "gru", model, config, models_config)

    assert trainer.loss_name == "focal"
    assert trainer.focal_gamma == 1.5
    trainer.pos_weight = 2.0
    trainer.criterion = trainer._build_criterion()
    assert isinstance(trainer.criterion, BinaryFocalLossWithLogits)
