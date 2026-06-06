from __future__ import annotations

import numpy as np

from channel_sim.dataset_generator import sample_params


def test_rayleigh_and_rician_are_both_sampled():
    rng = np.random.default_rng(42)
    types = [sample_params(rng)["channel_type"] for _ in range(200)]
    assert "rayleigh" in types
    assert "rician" in types


def test_rayleigh_has_explicit_indicator():
    rng = np.random.default_rng(7)
    params = sample_params(rng, overrides={"channel_type": "rayleigh"})
    assert params["channel_type"] == "rayleigh"
    assert params["is_rician"] == 0.0
    assert params["k_factor_db"] == 0.0
