"""Parameter normalization for dataset generation."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "is_rician",
    "fc",
    "velocity_mps",
    "num_paths",
    "k_factor_db",
    "antenna_spacing",
    "delay_spread",
    "bandwidth",
]

FEATURE_RANGES = {
    "is_rician": (0.0, 1.0),
    "fc": (2.4e9, 28e9),
    "velocity_mps": (0.0, 120.0 / 3.6),
    "num_paths": (4.0, 16.0),
    "k_factor_db": (0.0, 15.0),
    "antenna_spacing": (0.125, 0.5),
    "delay_spread": (50e-9, 500e-9),
    "bandwidth": (5e6, 20e6),
}


def normalize_value(name: str, value: float) -> float:
    low, high = FEATURE_RANGES[name]
    if high == low:
        return 0.0
    return (float(value) - low) / (high - low)


def denormalize_value(name: str, value: float) -> float:
    low, high = FEATURE_RANGES[name]
    return low + float(value) * (high - low)


def params_to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array([normalize_value(name, params[name]) for name in FEATURE_NAMES], dtype=np.float32)
