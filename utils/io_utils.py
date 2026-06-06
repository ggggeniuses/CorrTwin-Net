"""Experiment IO helpers."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


def fallback_path(path: str | Path) -> Path:
    out = Path(path)
    return out.with_name(f"{out.stem}_latest{out.suffix}")


def create_run_dir(model_name: str, base_dir: str | Path = "results/runs", run_name: str | None = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_name:
        name = run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{model_name}_{stamp}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except PermissionError:
        fallback_path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        try:
            out.write_text("", encoding="utf-8")
        except PermissionError:
            fallback_path(out).write_text("", encoding="utf-8")
        return
    try:
        handle = out.open("w", newline="", encoding="utf-8")
    except PermissionError:
        handle = fallback_path(out).open("w", newline="", encoding="utf-8")
    with handle as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_yaml(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except PermissionError:
        fallback_path(out).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def copy_config_snapshot(config_path: str | Path | None, run_dir: str | Path) -> None:
    if not config_path:
        return
    src = Path(config_path)
    if src.exists():
        shutil.copy2(src, Path(run_dir) / src.name)
