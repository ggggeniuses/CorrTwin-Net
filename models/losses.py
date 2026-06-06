"""Loss utilities."""

from __future__ import annotations

import torch


def mse_with_smoothness(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth_weight: float = 0.0,
) -> torch.Tensor:
    mse = torch.mean((y_pred - y_true) ** 2)
    if smooth_weight <= 0:
        return mse
    pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    true_diff = y_true[:, 1:] - y_true[:, :-1]
    smooth = torch.mean((pred_diff - true_diff) ** 2)
    return mse + smooth_weight * smooth
