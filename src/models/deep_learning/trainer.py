from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.preprocessing.sequence import SequenceDataset


@dataclass
class TrainingHistory:
    train_losses: list[float]
    validation_losses: list[float]
    best_validation_loss: float
    epochs_completed: int
    best_monitor_name: str
    best_monitor_value: float
    best_validation_precision: float
    best_validation_recall: float
    best_validation_f1: float
    pos_weight: float
    loss_name: str


def compute_pos_weight_from_targets(targets: np.ndarray | list[int] | list[float]) -> float:
    target_array = np.asarray(targets, dtype=int)
    positive_count = int(target_array.sum())
    negative_count = int(len(target_array) - positive_count)
    if positive_count <= 0:
        return 1.0
    return float(negative_count / positive_count)


class BinaryFocalLossWithLogits(nn.Module):
    """Binary focal loss that can optionally reuse BCE pos_weight handling."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        focal_weight = (1.0 - pt).pow(self.gamma)
        return (focal_weight * bce_loss).mean()


def tune_decision_threshold(
    probabilities: np.ndarray | list[float],
    targets: np.ndarray | list[int] | list[float],
    *,
    metric: str = "f1",
    beta: float = 2.0,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.01,
    min_threshold: float | None = None,
) -> dict[str, float]:
    if metric not in {"f1", "f_beta", "f2", "recall"}:
        raise ValueError("metric must be one of: f1, f_beta, f2, recall")
    probability_array = np.asarray(probabilities, dtype=float)
    target_array = np.asarray(targets, dtype=int)
    threshold_floor = float(start if min_threshold is None else min_threshold)
    threshold_floor = max(float(start), min(float(end), threshold_floor))
    best_result = {
        "threshold": float(threshold_floor),
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": -1.0,
        "score": float("-inf"),
        "metric": metric,
        "beta": float(beta),
        "min_threshold": float(threshold_floor),
    }

    threshold = float(threshold_floor)
    while threshold <= float(end) + 1e-12:
        predictions = (probability_array >= threshold).astype(int)
        precision = float(precision_score(target_array, predictions, zero_division=0))
        recall = float(recall_score(target_array, predictions, zero_division=0))
        f1_value = float(f1_score(target_array, predictions, zero_division=0))
        score_value = {
            "f1": f1_value,
            "f_beta": float(fbeta_score(target_array, predictions, beta=beta, zero_division=0)),
            "f2": float(fbeta_score(target_array, predictions, beta=2.0, zero_division=0)),
            "recall": recall,
        }[metric]
        if (
            score_value > best_result["score"]
            or (np.isclose(score_value, best_result["score"]) and f1_value > best_result["f1_score"])
            or (np.isclose(score_value, best_result["score"]) and np.isclose(f1_value, best_result["f1_score"]) and recall > best_result["recall"])
            or (
                np.isclose(score_value, best_result["score"])
                and np.isclose(f1_value, best_result["f1_score"])
                and np.isclose(recall, best_result["recall"])
                and precision > best_result["precision"]
            )
        ):
            best_result = {
                "threshold": float(round(threshold, 2)),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_value,
                "score": float(score_value),
                "metric": metric,
                "beta": float(beta),
                "min_threshold": float(threshold_floor),
            }
        threshold += float(step)

    return best_result


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
        early_stopping_monitor: str = "val_loss",
        early_stopping_mode: str = "min",
        use_pos_weight: bool = False,
        pos_weight_strategy: str = "neg_pos_ratio",
        loss_name: str = "bce",
        focal_gamma: float = 2.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if early_stopping_monitor not in {"val_loss", "val_f1", "val_recall"}:
            raise ValueError("early_stopping_monitor must be one of: val_loss, val_f1, val_recall")
        if early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be 'min' or 'max'")
        if pos_weight_strategy != "neg_pos_ratio":
            raise ValueError("Unsupported pos_weight_strategy")
        if loss_name not in {"bce", "focal"}:
            raise ValueError("loss_name must be one of: bce, focal")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")

        resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(resolved_device)
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        self.use_pos_weight = use_pos_weight
        self.pos_weight_strategy = pos_weight_strategy
        self.loss_name = loss_name
        self.focal_gamma = float(focal_gamma)
        self.pos_weight = 1.0
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)
        print(f"Kullanilan cihaz: {self.device}")

    def _build_criterion(self) -> nn.Module:
        pos_weight_tensor = None
        if self.use_pos_weight:
            pos_weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float32, device=self.device)
        if self.loss_name == "focal":
            return BinaryFocalLossWithLogits(gamma=self.focal_gamma, pos_weight=pos_weight_tensor)
        if pos_weight_tensor is not None:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        return nn.BCEWithLogitsLoss()

    def fit(self, train_data: SequenceDataset, validation_data: SequenceDataset) -> TrainingHistory:
        train_loader = self._build_dataloader(train_data, shuffle=True)
        validation_loader = self._build_dataloader(validation_data, shuffle=False)

        if self.use_pos_weight:
            self.pos_weight = compute_pos_weight_from_targets(train_data.targets)
        else:
            self.pos_weight = 1.0
        self.criterion = self._build_criterion()

        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_validation_loss = float("inf")
        best_monitor_value = float("inf") if self.early_stopping_mode == "min" else float("-inf")
        best_validation_precision = 0.0
        best_validation_recall = 0.0
        best_validation_f1 = 0.0
        best_state_dict = deepcopy(self.model.state_dict())
        best_epoch = 0
        patience_counter = 0

        for epoch_index in range(self.epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            validation_result = self._evaluate_loader(validation_loader, threshold=0.5)
            validation_loss = float(validation_result["loss"])
            validation_precision = float(validation_result["precision"])
            validation_recall = float(validation_result["recall"])
            validation_f1 = float(validation_result["f1_score"])
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            print(
                f"Epoch {epoch_index + 1}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {validation_loss:.4f} | "
                f"val_f1: {validation_f1:.4f} | "
                f"val_recall: {validation_recall:.4f}"
            )

            monitor_value = {
                "val_loss": validation_loss,
                "val_f1": validation_f1,
                "val_recall": validation_recall,
            }[self.early_stopping_monitor]

            is_better = (
                monitor_value < best_monitor_value
                if self.early_stopping_mode == "min"
                else monitor_value > best_monitor_value
            )

            if is_better:
                best_monitor_value = monitor_value
                best_validation_loss = validation_loss
                best_validation_precision = validation_precision
                best_validation_recall = validation_recall
                best_validation_f1 = validation_f1
                best_state_dict = deepcopy(self.model.state_dict())
                best_epoch = epoch_index + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping_enabled and patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch_index + 1} (best epoch: {best_epoch})")
                self.model.load_state_dict(best_state_dict)
                return TrainingHistory(
                    train_losses=train_losses,
                    validation_losses=validation_losses,
                    best_validation_loss=best_validation_loss,
                    epochs_completed=epoch_index + 1,
                    best_monitor_name=self.early_stopping_monitor,
                    best_monitor_value=float(best_monitor_value),
                    best_validation_precision=float(best_validation_precision),
                    best_validation_recall=float(best_validation_recall),
                    best_validation_f1=float(best_validation_f1),
                    pos_weight=float(self.pos_weight),
                    loss_name=self.loss_name,
                )

        self.model.load_state_dict(best_state_dict)
        return TrainingHistory(
            train_losses=train_losses,
            validation_losses=validation_losses,
            best_validation_loss=best_validation_loss,
            epochs_completed=len(train_losses),
            best_monitor_name=self.early_stopping_monitor,
            best_monitor_value=float(best_monitor_value),
            best_validation_precision=float(best_validation_precision),
            best_validation_recall=float(best_validation_recall),
            best_validation_f1=float(best_validation_f1),
            pos_weight=float(self.pos_weight),
            loss_name=self.loss_name,
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
        evaluation_result = self._evaluate_loader(self._build_dataloader(sequence_data, shuffle=False), threshold=threshold)
        return {
            "accuracy": float(evaluation_result["accuracy"]),
            "precision": float(evaluation_result["precision"]),
            "recall": float(evaluation_result["recall"]),
            "f1_score": float(evaluation_result["f1_score"]),
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

    def _evaluate_loader(self, data_loader: DataLoader, threshold: float = 0.5) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        all_probabilities: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                probabilities = torch.sigmoid(logits)

                batch_size = int(features.size(0))
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_examples += batch_size
                all_probabilities.append(probabilities.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

        if total_examples == 0:
            raise ValueError("Data loader produced no batches")

        probability_array = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.asarray([], dtype=float)
        target_array = np.concatenate(all_targets, axis=0).astype(int) if all_targets else np.asarray([], dtype=int)
        predictions = (probability_array >= threshold).astype(int)
        return {
            "loss": total_loss / total_examples,
            "accuracy": float(accuracy_score(target_array, predictions)),
            "precision": float(precision_score(target_array, predictions, zero_division=0)),
            "recall": float(recall_score(target_array, predictions, zero_division=0)),
            "f1_score": float(f1_score(target_array, predictions, zero_division=0)),
        }

    def _build_dataloader(self, sequence_data: SequenceDataset, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(sequence_data.features, dtype=torch.float32),
            torch.as_tensor(sequence_data.targets, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
