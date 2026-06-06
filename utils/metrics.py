"""Curve metrics."""

from __future__ import annotations

import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def relative_error(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + eps))


def curve_correlation(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    pred = y_pred.reshape(y_pred.shape[0], -1) if y_pred.ndim > 1 else y_pred[None, :]
    true = y_true.reshape(y_true.shape[0], -1) if y_true.ndim > 1 else y_true[None, :]
    scores = []
    for p, t in zip(pred, true):
        p = p - p.mean()
        t = t - t.mean()
        denom = np.linalg.norm(p) * np.linalg.norm(t)
        scores.append(0.0 if denom < eps else float(np.dot(p, t) / denom))
    return float(np.mean(scores))


def all_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y_pred, y_true),
        "mae": mae(y_pred, y_true),
        "relative_error": relative_error(y_pred, y_true),
        "curve_correlation": curve_correlation(y_pred, y_true),
    }


def lag0_error(y_pred: np.ndarray, y_true: np.ndarray, curve_points: int, include_ccf: bool) -> float:
    segments = 3 if include_ccf else 2
    pred = y_pred.reshape(-1, segments, curve_points)
    true = y_true.reshape(-1, segments, curve_points)
    return float(np.mean(np.abs(pred[:, :, 0] - true[:, :, 0])))


def out_of_range_fraction(y_pred: np.ndarray, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.mean((y_pred < low) | (y_pred > high)))


def total_variation(y: np.ndarray, curve_points: int, include_ccf: bool) -> float:
    segments = 3 if include_ccf else 2
    curves = y.reshape(-1, segments, curve_points)
    return float(np.mean(np.sum(np.abs(np.diff(curves, axis=-1)), axis=-1)))


def curve_total_variation(y: np.ndarray) -> float:
    return float(np.mean(np.sum(np.abs(np.diff(y, axis=-1)), axis=-1)))


def segmented_curve_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    curve_points: int,
    include_ccf: bool,
) -> dict[str, float]:
    names = ["acf", "ccf", "fcf"] if include_ccf else ["acf", "fcf"]
    metrics = {}
    for index, name in enumerate(names):
        start = index * curve_points
        end = start + curve_points
        pred_part = y_pred[..., start:end]
        true_part = y_true[..., start:end]
        metrics[f"{name}_mse"] = mse(pred_part, true_part)
        metrics[f"{name}_mae"] = mae(pred_part, true_part)
        metrics[f"{name}_corr"] = curve_correlation(pred_part, true_part)
        metrics[f"{name}_lag0_error"] = float(np.mean(np.abs(pred_part[..., 0] - true_part[..., 0])))
        metrics[f"{name}_pred_total_variation"] = curve_total_variation(pred_part)
        metrics[f"{name}_true_total_variation"] = curve_total_variation(true_part)
    metrics["overall_mse"] = mse(y_pred, y_true)
    metrics["overall_mae"] = mae(y_pred, y_true)
    metrics["overall_relative_error"] = relative_error(y_pred, y_true)
    metrics["overall_corr"] = curve_correlation(y_pred, y_true)
    metrics["lag0_error"] = lag0_error(y_pred, y_true, curve_points, include_ccf)
    metrics["out_of_range_fraction"] = out_of_range_fraction(y_pred)
    metrics["pred_total_variation"] = total_variation(y_pred, curve_points, include_ccf)
    metrics["true_total_variation"] = total_variation(y_true, curve_points, include_ccf)
    # Backward-compatible names used by older comparison scripts.
    metrics["mse"] = metrics["overall_mse"]
    metrics["mae"] = metrics["overall_mae"]
    metrics["relative_error"] = metrics["overall_relative_error"]
    metrics["curve_correlation"] = metrics["overall_corr"]
    return metrics
