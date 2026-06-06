from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.dataset_generator import sample_params, simulate_curves, simulate_statistical_curves
from experiments.common import device_from_arg, load_checkpoint
from experiments.run_evaluate_curves import build_model
from utils.normalization import params_to_vector
from utils.plotting import plot_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simulator and surrogate inference speed.")
    parser.add_argument("--model", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--num-realizations", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    results_root = ROOT / args.results_dir
    ckpt = load_checkpoint(results_root / "checkpoints" / f"{args.model}_best.pt", device)
    model = build_model(ckpt).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rng = np.random.default_rng(99)
    params = [sample_params(rng) for _ in range(args.runs)]

    start = time.perf_counter()
    for i, item in enumerate(params):
        simulate_curves(
            item,
            curve_points=int(ckpt["curve_points"]),
            include_ccf=bool(ckpt["include_ccf"]),
            seed=7000 + i,
        )
    single_realization_time = (time.perf_counter() - start) / args.runs

    start = time.perf_counter()
    for i, item in enumerate(params):
        simulate_statistical_curves(
            item,
            curve_points=int(ckpt["curve_points"]),
            include_ccf=bool(ckpt["include_ccf"]),
            num_realizations=args.num_realizations,
            seed=9000 + i,
        )
    ensemble_time = (time.perf_counter() - start) / args.runs

    x = np.stack([params_to_vector(item) for item in params]).astype(np.float32)
    xb = torch.from_numpy(x).to(device)
    with torch.no_grad():
        model(xb[: min(len(xb), 8)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batched_model_time = (time.perf_counter() - start) / args.runs

    one = xb[:1]
    single_runs = min(args.runs, 200)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(single_runs):
            model(one)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_model_time = (time.perf_counter() - start) / single_runs

    out_csv = results_root / "metrics" / "speed_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mode", "seconds_per_sample", "batch_size", "device"])
        writer.writeheader()
        writer.writerow({"method": "traditional_simulation", "mode": "single_realization", "seconds_per_sample": single_realization_time, "batch_size": 1, "device": "cpu"})
        writer.writerow({"method": "traditional_simulation", "mode": f"{args.num_realizations}_realization_ensemble", "seconds_per_sample": ensemble_time, "batch_size": 1, "device": "cpu"})
        writer.writerow({"method": args.model, "mode": "single_sample_latency", "seconds_per_sample": single_model_time, "batch_size": 1, "device": str(device)})
        writer.writerow({"method": args.model, "mode": "batched_throughput", "seconds_per_sample": batched_model_time, "batch_size": args.runs, "device": str(device)})
    with (results_root / "metrics" / "speed_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_realizations": args.num_realizations,
                "single_realization_simulator": single_realization_time,
                "ensemble_simulator": ensemble_time,
                "model_single_sample": single_model_time,
                "model_batched_per_sample": batched_model_time,
                "ensemble_vs_single_model_speedup": ensemble_time / max(single_model_time, 1e-12),
                "ensemble_vs_batched_model_speedup": ensemble_time / max(batched_model_time, 1e-12),
                "device": str(device),
                "batch_size": args.runs,
            },
            f,
            indent=2,
        )

    plot_bar(
        ["1-real sim", f"{args.num_realizations}-real sim", f"{args.model} single", f"{args.model} batch"],
        [single_realization_time, ensemble_time, single_model_time, batched_model_time],
        "Inference Speed Comparison",
        "seconds per sample",
        results_root / "figures" / "inference_speed_comparison.png",
    )
    print(f"single realization simulation seconds/sample: {single_realization_time:.8f}")
    print(f"{args.num_realizations}-realization ensemble simulation seconds/sample: {ensemble_time:.8f}")
    print(f"{args.model} single-sample seconds/sample: {single_model_time:.8f}")
    print(f"{args.model} batched seconds/sample: {batched_model_time:.8f}")


if __name__ == "__main__":
    main()
