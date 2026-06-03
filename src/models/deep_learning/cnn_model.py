from __future__ import annotations

import torch
from torch import nn


class CNNModel(nn.Module):
    """Temporal CNN that classifies a sequence from pooled convolution features."""

    def __init__(
        self,
        input_channels: int,
        num_filters: int,
        kernel_size: int,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if num_filters <= 0:
            raise ValueError("num_filters must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")

        self.input_channels = input_channels
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected 3D input tensor [batch, time, features], got shape {tuple(inputs.shape)}")
        if inputs.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} features/channels, got {int(inputs.shape[-1])}"
            )

        temporal_inputs = inputs.transpose(1, 2)
        features = self.conv(temporal_inputs)
        features = self.activation(features)
        pooled = self.pool(features).squeeze(-1)
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)
