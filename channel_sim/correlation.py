"""Correlation-function computation for channel responses.

CorrTwin-Net's formal target is normalized ensemble correlation magnitude:
`|E[x[n] conj(x[n+k])] / E[|x[n]|^2]|`.

For multi-realization labels, callers should accumulate complex sufficient
statistics first and take the magnitude only after ensemble aggregation. This
implements `|mean(R_i)|`/power-weighted ensemble correlation, not
`mean(|R_i|)`.
"""

from __future__ import annotations

import numpy as np


def correlation_sufficient_statistics(
    x: np.ndarray,
    num_lags: int,
    remove_mean: bool = False,
) -> tuple[np.ndarray, float]:
    """Return complex lag numerators and zero-lag average power."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if remove_mean:
        x = x - np.mean(x)

    numerators = np.zeros(num_lags, dtype=np.complex128)
    if x.size == 0:
        numerators[0] = 1.0
        return numerators, 1.0

    power = float(np.mean(np.abs(x) ** 2))
    if power <= 1e-14:
        numerators[0] = 1.0
        return numerators, 1.0

    max_lag = min(num_lags, x.size)
    for lag in range(max_lag):
        if lag == 0:
            numerators[lag] = np.mean(np.abs(x) ** 2)
        else:
            numerators[lag] = np.mean(x[:-lag] * np.conj(x[lag:]))
    return np.nan_to_num(numerators, nan=0.0, posinf=0.0, neginf=0.0), power


def finalize_correlation_magnitude(
    numerator_sum: np.ndarray,
    power_sum: float,
) -> np.ndarray:
    """Convert ensemble complex sufficient statistics into `|R|`."""
    curve = np.asarray(numerator_sum, dtype=np.complex128) / max(float(power_sum), 1e-14)
    magnitude = np.abs(np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0))
    magnitude /= max(float(magnitude[0]), 1e-14)
    return np.clip(magnitude, 0.0, 1.0).astype(np.float32)


def compute_raw_correlation(x: np.ndarray, num_lags: int = 128) -> np.ndarray:
    """Return normalized complex single-realization correlation."""
    numerators, power = correlation_sufficient_statistics(x, num_lags, remove_mean=False)
    return numerators / max(power, 1e-14)


def compute_covariance_correlation(
    x: np.ndarray,
    num_lags: int = 128,
    remove_mean: bool = True,
) -> np.ndarray:
    """Return normalized covariance-style complex single-realization correlation."""
    numerators, power = correlation_sufficient_statistics(x, num_lags, remove_mean=remove_mean)
    return numerators / max(power, 1e-14)


def compute_single_realization_correlation_magnitude(
    x: np.ndarray,
    num_lags: int = 128,
    remove_mean: bool = False,
) -> np.ndarray:
    """Return `|R|` for one realization. Do not use for ensemble labels."""
    numerators, power = correlation_sufficient_statistics(x, num_lags, remove_mean=remove_mean)
    return finalize_correlation_magnitude(numerators, power)


def compute_correlation_magnitude(
    x: np.ndarray,
    num_lags: int = 128,
    remove_mean: bool = False,
) -> np.ndarray:
    """Backward-compatible alias for single-realization correlation magnitude."""
    return compute_single_realization_correlation_magnitude(x, num_lags, remove_mean)


def compute_temporal_acf(h_time: np.ndarray, num_lags: int = 128) -> np.ndarray:
    return compute_single_realization_correlation_magnitude(h_time, num_lags, remove_mean=False)


def compute_frequency_fcf(H_freq: np.ndarray, num_deltas: int = 128) -> np.ndarray:
    return compute_single_realization_correlation_magnitude(H_freq, num_deltas, remove_mean=False)


def compute_spatial_ccf(h_space: np.ndarray, num_distances: int = 128) -> np.ndarray:
    return compute_single_realization_correlation_magnitude(h_space, num_distances, remove_mean=False)
