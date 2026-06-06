from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.correlation import compute_frequency_fcf, compute_spatial_ccf, compute_temporal_acf
from channel_sim.dataset_generator import sample_params, simulate_channel_responses
from utils.plotting import plot_curve_pair

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot example ACF, CCF, and FCF curves.")
    parser.add_argument("--curve-points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    params = sample_params(
        rng,
        overrides={
            "fc": 3.5e9,
            "velocity_mps": 60.0 / 3.6,
            "num_paths": 12,
            "k_factor_db": 6.0,
            "channel_type": "rician",
            "antenna_spacing": 0.5,
            "delay_spread": 300e-9,
            "bandwidth": 20e6,
        },
    )
    h_time, h_space, h_freq = simulate_channel_responses(
        params,
        seed=args.seed,
        add_noise=False,
    )

    acf = compute_temporal_acf(h_time, args.curve_points)
    ccf = compute_spatial_ccf(h_space, args.curve_points)
    fcf = compute_frequency_fcf(h_freq, args.curve_points)

    plot_curve_pair(acf, None, "Example Temporal |ACF|", ROOT / "results" / "figures" / "example_acf.png")
    plot_curve_pair(ccf, None, "Example Spatial |CCF|", ROOT / "results" / "figures" / "example_ccf.png")
    plot_curve_pair(fcf, None, "Example Frequency |FCF|", ROOT / "results" / "figures" / "example_fcf.png")
    print("saved example curves to results/figures")


if __name__ == "__main__":
    main()
