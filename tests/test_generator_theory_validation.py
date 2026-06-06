from __future__ import annotations

import numpy as np

from channel_sim.dataset_generator import simulate_statistical_curves
from channel_sim.doppler import C_LIGHT
from channel_sim.theory import isotropic_spatial_ccf, jakes_temporal_acf


def _norm(curve):
    curve = np.abs(np.asarray(curve))
    return (curve / max(float(curve[0]), 1e-12)).astype(np.float32)


def test_actual_generator_theory_validation_small():
    curve_points = 12
    fc = 3.5e9
    velocity = 60.0 / 3.6
    time_step = 1e-5
    bandwidth = 20e6
    num_frequency_samples = 256
    delay_spread = 150e-9
    params = {
        "channel_type": "rayleigh",
        "is_rician": 0.0,
        "fc": fc,
        "velocity_mps": velocity,
        "num_paths": 96,
        "k_factor_db": 0.0,
        "antenna_spacing": 0.025,
        "delay_spread": delay_spread,
        "bandwidth": bandwidth,
    }
    y = simulate_statistical_curves(
        params,
        curve_points=curve_points,
        include_ccf=True,
        num_realizations=96,
        seed=91,
        time_step=time_step,
        num_frequency_samples=num_frequency_samples,
    )
    acf = y[:curve_points]
    ccf = y[curve_points : 2 * curve_points]
    fcf = y[2 * curve_points :]

    wavelength = C_LIGHT / fc
    lags = np.arange(curve_points) * time_step
    distances = np.arange(curve_points) * 0.025 * wavelength
    offsets = np.arange(curve_points) * bandwidth / (num_frequency_samples - 1)
    acf_theory = _norm(jakes_temporal_acf(lags, velocity, fc))
    ccf_theory = _norm(isotropic_spatial_ccf(distances, wavelength))
    fcf_theory = _norm(1.0 / (1.0 + 1j * 2.0 * np.pi * offsets * delay_spread))

    assert np.mean((acf - acf_theory) ** 2) < 0.03
    assert np.mean((ccf - ccf_theory) ** 2) < 0.03
    assert np.mean((fcf - fcf_theory) ** 2) < 0.03
