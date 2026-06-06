"""Shared experiment helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_npz_dataset(data_dir: str | Path = "data") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, bool]:
    data_path = ROOT / data_dir
    arrays = {}
    labels = {}
    curve_points = 0
    include_ccf = False
    for split in ["train", "val", "test"]:
        pack = np.load(data_path / f"{split}.npz", allow_pickle=True)
        arrays[split] = pack["x"].astype(np.float32)
        labels[split] = pack["y"].astype(np.float32)
        curve_points = int(pack["curve_points"][0])
        include_ccf = bool(pack["include_ccf"][0])
    return arrays, labels, curve_points, include_ccf


def model_output_dim(curve_points: int, include_ccf: bool) -> int:
    return curve_points * (3 if include_ccf else 2)


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_json(path: str | Path, payload: dict) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except PermissionError:
        out.with_name(f"{out.stem}_latest{out.suffix}").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(ROOT / path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ROOT / path, map_location=device)
