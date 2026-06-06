"""MLP baseline surrogate."""

from __future__ import annotations

import torch
from torch import nn


class MLPSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 256,
        hidden_dims: list[int] | tuple[int, ...] = (128, 256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
