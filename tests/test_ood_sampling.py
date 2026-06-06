from __future__ import annotations

from channel_sim.dataset_generator import generate_dataset
from experiments.run_ood_generalization import audit_ood_dataset, coerce_overrides


def test_ood_sampling_audit_accepts_configured_scenarios():
    scenarios = {
        "velocity_ood": {"velocity_mps": 45.0},
        "k_factor_ood": {"channel_type": "rician", "k_factor_db": 18.0},
        "path_num_ood": {"num_paths": 24},
        "delay_spread_ood": {"delay_spread": 8.0e-7},
        "bandwidth_ood": {"bandwidth": 4.0e7},
    }
    for name, overrides in scenarios.items():
        x, _, params = generate_dataset(
            samples=4,
            curve_points=8,
            include_ccf=True,
            num_realizations=2,
            seed=123,
            overrides=coerce_overrides(overrides),
        )
        report = audit_ood_dataset(name, params, x)
        assert any(value is True for key, value in report.items() if key.startswith("all_"))
