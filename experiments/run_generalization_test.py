from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.dataset_generator import generate_dataset
from experiments.common import device_from_arg, load_checkpoint, save_json
from experiments.run_evaluate_curves import build_model
from utils.metrics import all_metrics
from utils.plotting import plot_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple out-of-range generalization tests.")
    parser.add_argument("--model", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    ckpt = load_checkpoint(f"results/checkpoints/{args.model}_best.pt", device)
    model = build_model(ckpt).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tests = {
        "velocity_high": {"velocity_mps": 110.0 / 3.6},
        "k_factor_high": {"k_factor_db": 14.0},
        "num_paths_16": {"num_paths": 16},
        "fc_28ghz": {"fc": 28e9},
    }
    results = {}
    for name, overrides in tests.items():
        x, y, _ = generate_dataset(
            samples=args.samples,
            curve_points=int(ckpt["curve_points"]),
            include_ccf=bool(ckpt["include_ccf"]),
            seed=123,
            overrides=overrides,
        )
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        results[name] = all_metrics(pred, y)

    save_json(f"results/metrics/{args.model}_generalization.json", results)
    labels = list(results.keys())
    values = [results[name]["mse"] for name in labels]
    plot_bar(
        labels,
        values,
        "Generalization MSE",
        "MSE",
        ROOT / "results" / "figures" / f"{args.model}_generalization_mse.png",
    )
    print(results)


if __name__ == "__main__":
    main()
