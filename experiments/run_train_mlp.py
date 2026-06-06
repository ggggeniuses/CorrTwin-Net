from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import device_from_arg, load_npz_dataset, model_output_dim, save_json
from models.mlp_surrogate import MLPSurrogate
from utils.io_utils import create_run_dir, save_csv, save_json as save_json_file, save_yaml
from utils.metrics import segmented_curve_metrics
from utils.plotting import plot_loss
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP surrogate.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x).to(device)
        pred = model(xb).cpu().numpy()
    return float(np.mean((pred - y) ** 2)), pred


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    device = device_from_arg(args.device)
    arrays, labels, curve_points, include_ccf = load_npz_dataset(args.data_dir)
    results_root = ROOT / args.results_dir
    output_dim = model_output_dim(curve_points, include_ccf)
    run_dir = create_run_dir("mlp", results_root / "runs", args.run_name)
    save_yaml(run_dir / "config.yaml", vars(args) | {"curve_points": curve_points, "include_ccf": include_ccf})

    model = MLPSurrogate(input_dim=arrays["train"].shape[1], output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(arrays["train"]), torch.from_numpy(labels["train"]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    history = []
    best_val = float("inf")
    best_epoch = 0
    ckpt_dir = results_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "mlp_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, _ = evaluate(model, arrays["val"], labels["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": "mlp",
                    "state_dict": model.state_dict(),
                    "input_dim": arrays["train"].shape[1],
                    "output_dim": output_dim,
                    "curve_points": curve_points,
                    "include_ccf": include_ccf,
                },
                best_ckpt_path,
            )
            torch.save(
                {
                    "model": "mlp",
                    "state_dict": model.state_dict(),
                    "input_dim": arrays["train"].shape[1],
                    "output_dim": output_dim,
                    "curve_points": curve_points,
                    "include_ccf": include_ccf,
                },
                run_dir / "model.pt",
            )
        print(f"epoch {epoch:03d} train={row['train_loss']:.6f} val={val_loss:.6f}")

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
    _, test_pred = evaluate(model, arrays["test"], labels["test"], device)
    metrics = segmented_curve_metrics(test_pred, labels["test"], curve_points, include_ccf)
    ckpt_sha256 = hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest() if best_ckpt_path.exists() else ""
    metrics.update({"best_epoch": best_epoch, "best_val_mse": best_val, "epochs": args.epochs, "checkpoint_sha256": ckpt_sha256})
    save_json_file(results_root / "metrics" / "mlp_metrics.json", metrics)
    save_json_file(run_dir / "metrics.json", metrics)
    plot_loss(history, results_root / "figures" / "mlp_loss.png")

    save_csv(results_root / "metrics" / "mlp_history.csv", history)
    save_csv(run_dir / "training_log.csv", history)
    print(f"test metrics: {metrics}")
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
