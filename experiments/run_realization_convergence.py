from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.dataset_generator import simulate_statistical_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study label convergence versus number of realizations.")
    parser.add_argument("--curve-points", type=int, default=64)
    parser.add_argument("--reference-realizations", type=int, default=2000)
    parser.add_argument("--realizations", type=str, default="1,2,4,8,16,32,64,100,200")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def representative_params() -> dict:
    return {
        "channel_type": "rayleigh",
        "is_rician": 0.0,
        "fc": 3.5e9,
        "velocity_mps": 60.0 / 3.6,
        "num_paths": 16,
        "k_factor_db": 0.0,
        "antenna_spacing": 0.25,
        "delay_spread": 150e-9,
        "bandwidth": 20e6,
    }


def main() -> None:
    args = parse_args()
    results_root = ROOT / args.results_dir
    metrics_dir = results_root / "metrics"
    figures_dir = results_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = representative_params()
    start = time.perf_counter()
    reference = simulate_statistical_curves(
        params,
        curve_points=args.curve_points,
        include_ccf=True,
        num_realizations=args.reference_realizations,
        seed=args.seed,
    )
    reference_time = time.perf_counter() - start

    rows = []
    for n in [int(v.strip()) for v in args.realizations.split(",") if v.strip()]:
        start = time.perf_counter()
        estimates = [
            simulate_statistical_curves(
                params,
                curve_points=args.curve_points,
                include_ccf=True,
                num_realizations=n,
                seed=args.seed + 10000 + rep * 97,
            )
            for rep in range(3)
        ]
        elapsed = (time.perf_counter() - start) / len(estimates)
        stack = np.stack(estimates)
        rows.append(
            {
                "num_realizations": n,
                "label_mse_vs_reference": float(np.mean((stack.mean(axis=0) - reference) ** 2)),
                "label_variance": float(np.mean(np.var(stack, axis=0))),
                "seconds_per_label": elapsed,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "realization_convergence.csv"
    json_path = metrics_dir / "realization_convergence.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "reference_realizations": args.reference_realizations,
                "reference_seconds": reference_time,
                "params": params,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(7, 4))
    plt.loglog(df["num_realizations"], df["label_mse_vs_reference"], marker="o", label="MSE vs reference")
    plt.loglog(df["num_realizations"], df["label_variance"], marker="s", label="label variance")
    plt.xlabel("num realizations")
    plt.ylabel("error")
    plt.title("Realization Convergence")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "realization_convergence.png", dpi=180)
    plt.close()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
