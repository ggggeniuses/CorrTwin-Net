from __future__ import annotations

from channel_sim.dataset_generator import generate_dataset


def test_dataset_generation_without_ccf():
    x, y, params = generate_dataset(samples=12, curve_points=32, include_ccf=False, seed=1)
    assert x.shape == (12, 8)
    assert y.shape == (12, 64)
    assert len(params) == 12


def test_dataset_generation_with_ccf():
    x, y, _ = generate_dataset(samples=12, curve_points=32, include_ccf=True, seed=2)
    assert x.shape == (12, 8)
    assert y.shape == (12, 96)
