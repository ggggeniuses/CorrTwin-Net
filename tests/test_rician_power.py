from __future__ import annotations

import numpy as np

from channel_sim.rician import sample_rician_paths


def test_rician_power_ratio_and_total_power():
    k_db = 6.0
    channel = sample_rician_paths(
        fc=3.5e9,
        velocity_mps=10.0,
        num_paths=12,
        k_factor_db=k_db,
        delay_spread=150e-9,
        seed=9,
    )
    los_power = float(np.abs(channel.gains[0]) ** 2)
    nlos_power = float(np.sum(np.abs(channel.gains[1:]) ** 2))
    k_linear = 10.0 ** (k_db / 10.0)

    assert np.isclose(los_power + nlos_power, 1.0, atol=1e-6)
    assert np.isclose(los_power / nlos_power, k_linear, rtol=1e-6)
