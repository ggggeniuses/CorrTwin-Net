from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import device_from_arg, load_checkpoint, load_npz_dataset
from models.mlp_surrogate import MLPSurrogate
from models.resmlp_surrogate import ResMLPSurrogate
from utils.metrics import segmented_curve_metrics
from utils.io_utils import save_json
from utils.plotting import plot_curve_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted correlation curves.")
    parser.add_argument("--model", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def build_model(ckpt: dict) -> torch.nn.Module:
    if ckpt["model"] == "mlp":
        return MLPSurrogate(ckpt["input_dim"], ckpt["output_dim"])
    return ResMLPSurrogate(
        ckpt["input_dim"],
        ckpt["output_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
        num_blocks=ckpt.get("num_blocks", 3),
    )


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    results_root = ROOT / args.results_dir
    arrays, labels, curve_points, include_ccf = load_npz_dataset(args.data_dir)
    ckpt = load_checkpoint(results_root / "checkpoints" / f"{args.model}_best.pt", device)
    model = build_model(ckpt).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(arrays["test"]).to(device)).cpu().numpy()
    metrics = segmented_curve_metrics(pred, labels["test"], curve_points, include_ccf)
    save_json(results_root / "metrics" / f"{args.model}_eval_metrics.json", metrics)

    idx = min(max(args.sample_index, 0), len(labels["test"]) - 1)
    true = labels["test"][idx]
    est = pred[idx]
    plot_curve_pair(
        true[:curve_points],
        est[:curve_points],
        "Temporal ACF Prediction",
        results_root / "figures" / f"{args.model}_acf_prediction_example.png",
    )
    if include_ccf:
        plot_curve_pair(
            true[curve_points : 2 * curve_points],
            est[curve_points : 2 * curve_points],
            "Spatial CCF Prediction",
            results_root / "figures" / f"{args.model}_ccf_prediction_example.png",
        )
        fcf_start = 2 * curve_points
    else:
        fcf_start = curve_points
    plot_curve_pair(
        true[fcf_start : fcf_start + curve_points],
        est[fcf_start : fcf_start + curve_points],
        "Frequency FCF Prediction",
        results_root / "figures" / f"{args.model}_fcf_prediction_example.png",
    )
    print(f"evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
