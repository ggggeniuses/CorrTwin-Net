"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _savefig(output_path: str | Path, dpi: int = 180) -> None:
    path = Path(output_path)
    try:
        plt.savefig(path, dpi=dpi)
    except PermissionError:
        plt.savefig(path.with_name(f"{path.stem}_latest{path.suffix}"), dpi=dpi)


def plot_curve_pair(
    true_curve: np.ndarray,
    pred_curve: np.ndarray | None,
    title: str,
    output_path: str | Path,
) -> None:
    ensure_parent(output_path)
    plt.figure(figsize=(7, 4))
    plt.plot(true_curve, label="simulation", linewidth=2)
    if pred_curve is not None:
        plt.plot(pred_curve, label="surrogate", linewidth=2, linestyle="--")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("lag / delta index")
    plt.ylabel("normalized correlation")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _savefig(output_path, dpi=180)
    plt.close()


def plot_loss(history: list[dict[str, float]], output_path: str | Path) -> None:
    ensure_parent(output_path)
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _savefig(output_path, dpi=180)
    plt.close()


def plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, output_path: str | Path) -> None:
    ensure_parent(output_path)
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _savefig(output_path, dpi=180)
    plt.close()
