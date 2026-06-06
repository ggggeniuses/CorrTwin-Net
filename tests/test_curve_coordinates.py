from __future__ import annotations

import numpy as np

from channel_sim.dataset_generator import curve_coordinates_for_params


def test_curve_coordinates_are_sample_specific():
    params = {
        "antenna_spacing": 0.25,
        "bandwidth": 10e6,
    }
    coords = curve_coordinates_for_params(params, curve_points=5, num_frequency_samples=101, time_step=2e-4)

    assert np.array_equal(coords["lag_indices"], np.arange(5))
    assert np.allclose(coords["time_lags_s"], np.arange(5) * 2e-4)
    assert np.allclose(coords["spatial_distances_lambda"], np.arange(5) * 0.25)
    assert np.allclose(coords["frequency_offsets_hz"], np.arange(5) * 10e6 / 100)
