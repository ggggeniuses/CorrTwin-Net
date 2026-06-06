"""Doppler and wavelength helpers."""

from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0


def kmh_to_mps(speed_kmh: float) -> float:
    return speed_kmh / 3.6


def wavelength(fc: float) -> float:
    if fc <= 0:
        raise ValueError("Carrier frequency must be positive.")
    return C_LIGHT / fc


def max_doppler_shift(fc: float, velocity_mps: float) -> float:
    return abs(velocity_mps) / C_LIGHT * fc


def path_doppler_shifts(
    fc: float,
    velocity_mps: float,
    angles_rad: np.ndarray,
) -> np.ndarray:
    fd_max = max_doppler_shift(fc, velocity_mps)
    return fd_max * np.cos(angles_rad)
