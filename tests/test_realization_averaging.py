from __future__ import annotations

import numpy as np

from channel_sim.dataset_generator import sample_params, simulate_statistical_curves


def test_realization_averaging_reduces_seed_variation():
    rng = np.random.default_rng(5)
    params = sample_params(rng, overrides={"channel_type": "rayleigh"})
    y1 = simulate_statistical_curves(params, curve_points=32, include_ccf=True, num_realizations=2, seed=10)
    y2 = simulate_statistical_curves(params, curve_points=32, include_ccf=True, num_realizations=2, seed=20)
    z1 = simulate_statistical_curves(params, curve_points=32, include_ccf=True, num_realizations=16, seed=10)
    z2 = simulate_statistical_curves(params, curve_points=32, include_ccf=True, num_realizations=16, seed=20)
    assert np.mean(np.abs(z1 - z2)) < np.mean(np.abs(y1 - y2))
