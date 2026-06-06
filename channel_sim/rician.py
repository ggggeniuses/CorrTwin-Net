"""Simplified Rician multipath channel generation."""

from __future__ import annotations

import numpy as np

from .doppler import path_doppler_shifts
from .rayleigh import MultipathChannel, sample_rayleigh_paths


def sample_rician_paths(
    fc: float,
    velocity_mps: float,
    num_paths: int,
    k_factor_db: float,
    delay_spread: float,
    seed: int | None = None,
) -> MultipathChannel:
    """Sample a Rician channel with one LoS path and diffuse NLoS paths."""
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1.")

    rng = np.random.default_rng(seed)
    k_linear = 10.0 ** (k_factor_db / 10.0)
    los_power = k_linear / (k_linear + 1.0)
    nlos_power = 1.0 / (k_linear + 1.0)

    if num_paths == 1:
        angles = np.array([0.0])
        delays = np.array([0.0])
        gains = np.array([1.0 + 0.0j], dtype=np.complex128)
    else:
        nlos = sample_rayleigh_paths(
            fc=fc,
            velocity_mps=velocity_mps,
            num_paths=num_paths - 1,
            delay_spread=delay_spread,
            seed=None if seed is None else seed + 17,
        )
        los_phase = rng.uniform(0.0, 2.0 * np.pi)
        los_gain = np.sqrt(los_power) * np.exp(1j * los_phase)
        nlos_gains = nlos.gains.astype(np.complex128)
        nlos_norm = max(float(np.sqrt(np.sum(np.abs(nlos_gains) ** 2))), 1e-14)
        nlos_gains = nlos_gains / nlos_norm
        angles = np.concatenate([np.array([0.0]), nlos.angles_rad])
        delays = np.concatenate([np.array([0.0]), nlos.delays_s])
        gains = np.concatenate([np.array([los_gain]), np.sqrt(nlos_power) * nlos_gains])

    dopplers = path_doppler_shifts(fc, velocity_mps, angles)
    return MultipathChannel(gains, angles, delays, dopplers, fc)
