from __future__ import annotations

import numpy as np

from channel_sim.dataset_generator import simulate_statistical_curves


def _theory(offsets: np.ndarray, tau_rms: float) -> np.ndarray:
    curve = np.abs(1.0 / (1.0 + 1j * 2.0 * np.pi * offsets * tau_rms))
    return (curve / curve[0]).astype(np.float32)


def test_actual_generator_fcf_matches_exponential_pdp_reasonably():
    curve_points = 16
    bandwidth = 20e6
    num_frequency_samples = 256
    tau = 120e-9
    params = {
        "channel_type": "rayleigh",
        "is_rician": 0.0,
        "fc": 3.5e9,
        "velocity_mps": 0.0,
        "num_paths": 96,
        "k_factor_db": 0.0,
        "antenna_spacing": 0.25,
        "delay_spread": tau,
        "bandwidth": bandwidth,
    }
    y = simulate_statistical_curves(
        params,
        curve_points=curve_points,
        include_ccf=True,
        num_realizations=96,
        seed=77,
        num_frequency_samples=num_frequency_samples,
    )
    fcf = y[2 * curve_points :]
    offsets = np.arange(curve_points) * bandwidth / (num_frequency_samples - 1)

    assert np.mean((fcf - _theory(offsets, tau)) ** 2) < 0.01
