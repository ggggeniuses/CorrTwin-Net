from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.dataset_generator import generate_dataset, save_dataset_metadata, save_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CorrTwin-Net dataset.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--curve-points", type=int, default=128)
    parser.add_argument("--include-ccf", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--num-realizations", type=int, default=16)
    parser.add_argument("--num-time-samples", type=int, default=512)
    parser.add_argument("--num-frequency-samples", type=int, default=512)
    parser.add_argument("--num-space-samples", type=int, default=512)
    parser.add_argument("--time-step", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, params = generate_dataset(
        samples=args.samples,
        curve_points=args.curve_points,
        include_ccf=args.include_ccf,
        seed=args.seed,
        num_realizations=args.num_realizations,
        num_time_samples=args.num_time_samples,
        num_frequency_samples=args.num_frequency_samples,
        num_space_samples=args.num_space_samples,
        time_step=args.time_step,
    )
    save_splits(
        x,
        y,
        params,
        ROOT / args.output_dir,
        args.curve_points,
        args.include_ccf,
        num_realizations=args.num_realizations,
        seed=args.seed,
        num_time_samples=args.num_time_samples,
        num_frequency_samples=args.num_frequency_samples,
        num_space_samples=args.num_space_samples,
        time_step=args.time_step,
    )
    save_dataset_metadata(
        ROOT / args.output_dir,
        samples=args.samples,
        curve_points=args.curve_points,
        include_ccf=args.include_ccf,
        num_realizations=args.num_realizations,
        seed=args.seed,
        num_time_samples=args.num_time_samples,
        num_frequency_samples=args.num_frequency_samples,
        num_space_samples=args.num_space_samples,
        time_step=args.time_step,
    )
    print(f"saved dataset to {ROOT / args.output_dir}")
    print(f"x shape: {x.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    main()
