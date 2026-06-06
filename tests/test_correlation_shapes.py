from __future__ import annotations

import numpy as np

from channel_sim.correlation import compute_frequency_fcf, compute_spatial_ccf, compute_temporal_acf


def test_correlation_shapes_and_normalization():
    rng = np.random.default_rng(7)
    x = rng.normal(size=128) + 1j * rng.normal(size=128)
    for fn in [compute_temporal_acf, compute_frequency_fcf, compute_spatial_ccf]:
        curve = fn(x, 32)
        assert curve.shape == (32,)
        assert np.all(np.isfinite(curve))
        assert np.isclose(curve[0], 1.0)
        assert np.min(curve) >= 0.0
        assert np.max(curve) <= 1.0
