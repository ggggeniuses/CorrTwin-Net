from __future__ import annotations

import warnings

import numpy as np

from channel_sim.correlation import finalize_correlation_magnitude


def test_finalize_correlation_magnitude_warns_instead_of_clipping() -> None:
    numerator = np.array([1.0 + 0.0j, 1.08 + 0.0j], dtype=np.complex128)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        curve = finalize_correlation_magnitude(numerator, power_sum=1.0)

    assert curve[1] > 1.0
    assert any("unclipped" in str(item.message) for item in caught)
