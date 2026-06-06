from __future__ import annotations

import json
from pathlib import Path


def test_formal_actual_generator_theory_metrics() -> None:
    metrics_path = Path("results_formal_p0/metrics/theory_validation.json")
    assert metrics_path.exists(), "formal theory_validation.json is required for release audit"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual = metrics["actual_generator"]
    for name in ["acf", "ccf", "fcf"]:
        assert actual[f"{name}_mse"] < 0.01
        assert actual[f"{name}_max_abs_error"] < 0.15
    assert actual["max_abs_error"] < 0.15
