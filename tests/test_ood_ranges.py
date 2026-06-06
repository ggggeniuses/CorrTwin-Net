from __future__ import annotations

from pathlib import Path

import yaml

from utils.normalization import FEATURE_RANGES


def test_path_num_ood_is_outside_training_range():
    config = yaml.safe_load(Path("configs/ood_generalization.yaml").read_text(encoding="utf-8"))
    value = config["scenarios"]["path_num_ood"]["num_paths"]
    assert value > FEATURE_RANGES["num_paths"][1]
