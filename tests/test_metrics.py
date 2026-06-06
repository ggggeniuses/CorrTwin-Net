from __future__ import annotations

import numpy as np

from utils.metrics import curve_correlation, mae, mse, relative_error, segmented_curve_metrics


def test_basic_metrics():
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])
    assert np.isclose(mse(pred, true), 1.0 / 3.0)
    assert np.isclose(mae(pred, true), 1.0 / 3.0)
    assert relative_error(pred, true) > 0


def test_curve_correlation_identical():
    y = np.array([[1.0, 0.5, 0.25]])
    assert np.isclose(curve_correlation(y, y), 1.0)


def test_segmented_metrics_keys():
    y = np.ones((2, 96))
    metrics = segmented_curve_metrics(y, y, curve_points=32, include_ccf=True)
    for key in ["acf_mse", "ccf_mse", "fcf_mse", "overall_mse", "overall_corr"]:
        assert key in metrics
    assert metrics["overall_mse"] == 0.0
