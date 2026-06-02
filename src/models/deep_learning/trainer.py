from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing.sequence import SequenceDataset


@dataclass
class TrainingHistory:
    train_losses: list[float]
    validation_losses: list[float]
    best_validation_loss: float
    epochs_completed: int


class Trainer:
    """Reusable training loop for binary sequence classifiers."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        device: str = "cpu",
        early_stopping_enabled: bool = True,
        early_stopping_patience: int = 5,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_patience = early_stopping_patience
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def fit(self, train_data: SequenceDataset, validation_data: SequenceDataset) -> TrainingHistory:
        train_loader = self._build_dataloader(train_data, shuffle=True)
        validation_loader = self._build_dataloader(validation_data, shuffle=False)

        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_validation_loss = float("inf")
        best_state_dict = deepcopy(self.model.state_dict())
        patience_counter = 0

        for epoch_index in range(self.epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            validation_loss = self._run_epoch(validation_loader, training=False)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state_dict = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping_enabled and patience_counter >= self.early_stopping_patience:
                self.model.load_state_dict(best_state_dict)
                return TrainingHistory(
                    train_losses=train_losses,
                    validation_losses=validation_losses,
                    best_validation_loss=best_validation_loss,
                    epochs_completed=epoch_index + 1,
                )

        self.model.load_state_dict(best_state_dict)
        return TrainingHistory(
            train_losses=train_losses,
            validation_losses=validation_losses,
            best_validation_loss=best_validation_loss,
            epochs_completed=len(train_losses),
        )

    def predict_logits(self, sequence_data: SequenceDataset) -> np.ndarray:
        data_loader = self._build_dataloader(sequence_data, shuffle=False)
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in data_loader:
                logits = self.model(features.to(self.device))
                predictions.append(logits.detach().cpu().numpy())
        return np.concatenate(predictions, axis=0) if predictions else np.asarray([], dtype=float)

    def predict_probabilities(self, sequence_data: SequenceDataset) -> np.ndarray:
        logits = self.predict_logits(sequence_data)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_labels(self, sequence_data: SequenceDataset, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_probabilities(sequence_data)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, sequence_data: SequenceDataset, threshold: float = 0.5) -> dict[str, float]:
        probabilities = self.predict_probabilities(sequence_data)
        predictions = (probabilities >= threshold).astype(int)
        targets = sequence_data.targets.astype(int)
        return {
            "accuracy": float(accuracy_score(targets, predictions)),
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "recall": float(recall_score(targets, predictions, zero_division=0)),
            "f1_score": float(f1_score(targets, predictions, zero_division=0)),
        }

    def _run_epoch(self, data_loader: DataLoader, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_examples = 0

        for features, targets in data_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                if training:
                    loss.backward()
                    self.optimizer.step()

            batch_size = int(features.size(0))
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size

        if total_examples == 0:
            raise ValueError("Data loader produced no batches")
        return total_loss / total_examples

    def _build_dataloader(self, sequence_data: SequenceDataset, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(sequence_data.features, dtype=torch.float32),
            torch.as_tensor(sequence_data.targets, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
