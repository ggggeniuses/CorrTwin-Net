"""YAML config loading and command-line merge helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml


def load_yaml_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config_args(config: dict, args: Namespace) -> dict:
    merged = dict(config)
    for key, value in vars(args).items():
        if value is not None:
            merged[key.replace("-", "_")] = value
    return merged
