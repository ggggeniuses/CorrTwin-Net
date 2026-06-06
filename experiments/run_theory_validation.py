from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.doppler import C_LIGHT
from channel_sim.dataset_generator import simulate_statistical_curves
from channel_sim.theory import isotropic_spatial_ccf, jakes_temporal_acf
from utils.io_utils import save_json
from utils.metrics import mae, mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Monte Carlo correlation magnitudes against theory.")
    parser.add_argument("--curve-points", type=int, default=128)
    parser.add_argument("--fc", type=float, default=3.5e9)
    parser.add_argument("--velocity", type=float, default=60.0 / 3.6)
    parser.add_argument("--num-realizations", type=int, default=4096)
    parser.add_argument("--time-step", type=float, default=1e-5)
    parser.add_argument("--delay-spread", type=float, default=150e-9)
    parser.add_argument("--bandwidth", type=float, default=20e6)
    parser.add_argument("--num-paths", type=int, default=64)
    parser.add_argument("--num-frequency-samples", type=int, default=512)
    parser.add_argument("--validate-actual-generator", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def normalize_magnitude(curve: np.ndarray) -> np.ndarray:
    curve = np.abs(np.nan_to_num(curve))
    return np.clip(curve / max(float(curve[0]), 1e-12), 0.0, 1.0).astype(np.float32)


def monte_carlo_temporal_acf(lags: np.ndarray, velocity: float, fc: float, n: int, rng: np.random.Generator) -> np.ndarray:
    fd = velocity * fc / C_LIGHT
    angles = rng.uniform(-np.pi, np.pi, size=n)
    phase0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    h0 = np.exp(1j * phase0)
    curves = []
    for tau in lags:
        h_tau = np.exp(1j * (phase0 + 2.0 * np.pi * fd * np.cos(angles) * tau))
        curves.append(np.mean(h0 * np.conj(h_tau)) / np.mean(np.abs(h0) ** 2))
    return normalize_magnitude(np.asarray(curves))


def monte_carlo_spatial_ccf(distances: np.ndarray, wavelength: float, n: int, rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(-np.pi, np.pi, size=n)
    phase0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    h0 = np.exp(1j * phase0)
    curves = []
    for distance in distances:
        h_d = np.exp(1j * (phase0 + 2.0 * np.pi * np.sin(angles) * distance / wavelength))
        curves.append(np.mean(h0 * np.conj(h_d)) / np.mean(np.abs(h0) ** 2))
    return normalize_magnitude(np.asarray(curves))


def exponential_pdp_fcf(offsets: np.ndarray, delay_spread: float) -> np.ndarray:
    return normalize_magnitude(1.0 / (1.0 + 1j * 2.0 * np.pi * offsets * delay_spread))


def monte_carlo_frequency_fcf(offsets: np.ndarray, delay_spread: float, n: int, rng: np.random.Generator) -> np.ndarray:
    delays = rng.exponential(scale=delay_spread, size=n)
    weights = np.ones(n, dtype=np.complex128) / n
    curves = []
    for offset in offsets:
        curves.append(np.sum(weights * np.exp(-1j * 2.0 * np.pi * offset * delays)))
    return normalize_magnitude(np.asarray(curves))


def plot_validation(x: np.ndarray, sim: np.ndarray, theory: np.ndarray, title: str, xlabel: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(x, theory, label="theory", linewidth=2)
    plt.plot(x, sim, label="simulation ensemble", linewidth=2, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("normalized correlation magnitude")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(output, dpi=180)
    except PermissionError:
        plt.savefig(output.with_name(f"{output.stem}_latest{output.suffix}"), dpi=180)
    plt.close()


def metric_block(acf_sim: np.ndarray, acf_theory: np.ndarray, ccf_sim: np.ndarray, ccf_theory: np.ndarray, fcf_sim: np.ndarray, fcf_theory: np.ndarray) -> dict[str, float]:
    return {
        "acf_mse": mse(acf_sim, acf_theory),
        "acf_mae": mae(acf_sim, acf_theory),
        "acf_max_abs_error": float(np.max(np.abs(acf_sim - acf_theory))),
        "ccf_mse": mse(ccf_sim, ccf_theory),
        "ccf_mae": mae(ccf_sim, ccf_theory),
        "ccf_max_abs_error": float(np.max(np.abs(ccf_sim - ccf_theory))),
        "fcf_mse": mse(fcf_sim, fcf_theory),
        "fcf_mae": mae(fcf_sim, fcf_theory),
        "fcf_max_abs_error": float(np.max(np.abs(fcf_sim - fcf_theory))),
        "max_abs_error": float(
            max(
                np.max(np.abs(acf_sim - acf_theory)),
                np.max(np.abs(ccf_sim - ccf_theory)),
                np.max(np.abs(fcf_sim - fcf_theory)),
            )
        ),
    }


def main() -> None:
    args = parse_args()
    results_root = ROOT / args.results_dir
    rng = np.random.default_rng(args.seed)
    wavelength = C_LIGHT / args.fc

    lags = np.arange(args.curve_points, dtype=np.float64) * args.time_step
    acf_sim = monte_carlo_temporal_acf(lags, args.velocity, args.fc, args.num_realizations, rng)
    acf_theory = normalize_magnitude(jakes_temporal_acf(lags, args.velocity, args.fc))

    distances = np.arange(args.curve_points, dtype=np.float64) * 0.025 * wavelength
    ccf_sim = monte_carlo_spatial_ccf(distances, wavelength, args.num_realizations, rng)
    ccf_theory = normalize_magnitude(isotropic_spatial_ccf(distances, wavelength))

    offsets = np.arange(args.curve_points, dtype=np.float64) * (args.bandwidth / max(args.num_frequency_samples - 1, 1))
    fcf_sim = monte_carlo_frequency_fcf(offsets, args.delay_spread, args.num_realizations, rng)
    fcf_theory = exponential_pdp_fcf(offsets, args.delay_spread)

    plot_validation(lags, acf_sim, acf_theory, "Temporal |ACF|: Ensemble vs Jakes Theory", "time lag (s)", results_root / "figures" / "theory_acf_validation.png")
    plot_validation(distances / wavelength, ccf_sim, ccf_theory, "Spatial |CCF|: Ensemble vs Isotropic Theory", "distance / wavelength", results_root / "figures" / "theory_ccf_validation.png")
    plot_validation(offsets / 1e6, fcf_sim, fcf_theory, "Frequency |FCF|: Ensemble vs Exponential PDP Theory", "frequency offset (MHz)", results_root / "figures" / "theory_fcf_validation.png")

    metrics = {
        "num_realizations": args.num_realizations,
        "analytic_monte_carlo": metric_block(acf_sim, acf_theory, ccf_sim, ccf_theory, fcf_sim, fcf_theory),
        "label_definition": "normalized ensemble correlation magnitude |R|",
    }

    if args.validate_actual_generator:
        actual_params = {
            "channel_type": "rayleigh",
            "is_rician": 0.0,
            "fc": args.fc,
            "velocity_mps": args.velocity,
            "num_paths": args.num_paths,
            "k_factor_db": 0.0,
            "antenna_spacing": 0.025,
            "delay_spread": args.delay_spread,
            "bandwidth": args.bandwidth,
        }
        actual = simulate_statistical_curves(
            actual_params,
            curve_points=args.curve_points,
            include_ccf=True,
            num_realizations=args.num_realizations,
            seed=args.seed + 900,
            time_step=args.time_step,
            num_frequency_samples=args.num_frequency_samples,
        )
        cp = args.curve_points
        actual_acf = actual[:cp]
        actual_ccf = actual[cp : 2 * cp]
        actual_fcf = actual[2 * cp :]
        actual_distances = np.arange(args.curve_points, dtype=np.float64) * 0.025 * wavelength
        actual_ccf_theory = normalize_magnitude(isotropic_spatial_ccf(actual_distances, wavelength))
        metrics["actual_generator"] = metric_block(
            actual_acf,
            acf_theory,
            actual_ccf,
            actual_ccf_theory,
            actual_fcf,
            fcf_theory,
        )
        plot_validation(lags, actual_acf, acf_theory, "Temporal |ACF|: Actual Generator vs Jakes Theory", "time lag (s)", results_root / "figures" / "theory_actual_acf_validation.png")
        plot_validation(actual_distances / wavelength, actual_ccf, actual_ccf_theory, "Spatial |CCF|: Actual Generator vs Isotropic Theory", "distance / wavelength", results_root / "figures" / "theory_actual_ccf_validation.png")
        plot_validation(offsets / 1e6, actual_fcf, fcf_theory, "Frequency |FCF|: Actual Generator vs Exponential PDP Theory", "frequency offset (MHz)", results_root / "figures" / "theory_actual_fcf_validation.png")
    save_json(results_root / "metrics" / "theory_validation.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
