from __future__ import annotations

import torch
from torch import nn


class LSTMModel(nn.Module):
    """Sequence classifier that predicts anomaly logits from the last hidden state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected 3D input tensor [batch, time, features], got shape {tuple(inputs.shape)}")

        _, (hidden_state, _) = self.lstm(inputs)
        final_hidden_state = hidden_state[-1]
        logits = self.classifier(self.dropout(final_hidden_state))
        return logits.squeeze(-1)
