from __future__ import annotations

import numpy as np

from channel_sim.doppler import wavelength
from channel_sim.theory import isotropic_spatial_ccf, jakes_temporal_acf


def test_theoretical_acf_lag_zero_is_one():
    curve = jakes_temporal_acf(np.array([0.0, 1e-4]), velocity=10.0, fc=3.5e9)
    assert np.isclose(curve[0], 1.0)


def test_theoretical_ccf_zero_distance_is_one():
    curve = isotropic_spatial_ccf(np.array([0.0, 0.5 * wavelength(3.5e9)]), wavelength(3.5e9))
    assert np.isclose(curve[0], 1.0)
    assert np.any(curve < 0.0)
