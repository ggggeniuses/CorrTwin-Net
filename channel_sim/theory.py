"""Lightweight theoretical correlation-function references."""

from __future__ import annotations

import numpy as np
from scipy.special import j0

from .doppler import C_LIGHT


def jakes_temporal_acf(lags: np.ndarray, velocity: float, fc: float, c: float = C_LIGHT) -> np.ndarray:
    """Classical Clarke/Jakes normalized temporal ACF."""
    lags = np.asarray(lags, dtype=np.float64)
    fd = abs(velocity) * fc / c
    curve = j0(2.0 * np.pi * fd * lags)
    return np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)


def isotropic_spatial_ccf(distances: np.ndarray, wavelength: float) -> np.ndarray:
    """Isotropic 2D arrival-angle spatial CCF sanity-check curve."""
    distances = np.asarray(distances, dtype=np.float64)
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    curve = j0(2.0 * np.pi * distances / wavelength)
    curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
    if curve.size:
        curve[0] = 1.0
    return curve
