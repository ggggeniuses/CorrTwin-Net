from __future__ import annotations

import numpy as np

from channel_sim.correlation import finalize_correlation_magnitude


def test_complex_ensemble_aggregation_takes_magnitude_last():
    r1 = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    r2 = np.array([1.0 + 0.0j, 0.0 - 1.0j])

    result = finalize_correlation_magnitude(r1 + r2, power_sum=2.0)
    mean_abs = (np.abs(r1) + np.abs(r2)) / 2.0

    assert np.allclose(result, np.array([1.0, 0.0], dtype=np.float32))
    assert not np.allclose(result, mean_abs.astype(np.float32))
