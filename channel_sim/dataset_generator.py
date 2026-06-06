"""Dataset generation for CorrTwin-Net."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .correlation import (
    correlation_sufficient_statistics,
    finalize_correlation_magnitude,
)
from .rayleigh import (
    add_awgn,
    sample_rayleigh_paths,
    synthesize_frequency_response,
    synthesize_spatial_response,
    synthesize_time_response,
)
from .rician import sample_rician_paths
from utils.normalization import FEATURE_NAMES, params_to_vector

FC_CHOICES = np.array([2.4e9, 3.5e9, 5.9e9, 28e9], dtype=np.float64)
NUM_PATH_CHOICES = np.array([4, 8, 12, 16], dtype=np.int64)
SPACING_CHOICES = np.array([0.125, 0.25, 0.5], dtype=np.float64)
BANDWIDTH_CHOICES = np.array([5e6, 10e6, 20e6], dtype=np.float64)


def sample_params(rng: np.random.Generator, overrides: dict[str, float | str] | None = None) -> dict[str, float | str]:
    channel_type = str(rng.choice(np.array(["rayleigh", "rician"])))
    is_rician = 1.0 if channel_type == "rician" else 0.0
    params: dict[str, float | str] = {
        "channel_type": channel_type,
        "is_rician": is_rician,
        "fc": float(rng.choice(FC_CHOICES)),
        "velocity_mps": float(rng.uniform(0.0, 120.0 / 3.6)),
        "num_paths": int(rng.choice(NUM_PATH_CHOICES)),
        "k_factor_db": float(rng.uniform(0.0, 15.0)) if is_rician else 0.0,
        "antenna_spacing": float(rng.choice(SPACING_CHOICES)),
        "delay_spread": float(rng.uniform(50e-9, 500e-9)),
        "bandwidth": float(rng.choice(BANDWIDTH_CHOICES)),
    }
    if overrides:
        params.update(overrides)
        if "channel_type" in overrides:
            params["channel_type"] = str(params["channel_type"])
            params["is_rician"] = 1.0 if params["channel_type"] == "rician" else 0.0
        elif "is_rician" in overrides:
            params["is_rician"] = float(params["is_rician"])
            params["channel_type"] = "rician" if float(params["is_rician"]) >= 0.5 else "rayleigh"
        if params["channel_type"] == "rayleigh":
            params["k_factor_db"] = 0.0
    return params


def _sample_channel(params: dict[str, float | str], seed: int | None = None):
    channel_type = str(params.get("channel_type", "rician" if float(params.get("is_rician", 1.0)) >= 0.5 else "rayleigh"))
    if channel_type == "rayleigh":
        return sample_rayleigh_paths(
            fc=float(params["fc"]),
            velocity_mps=float(params["velocity_mps"]),
            num_paths=int(params["num_paths"]),
            delay_spread=float(params["delay_spread"]),
            seed=seed,
        )
    return sample_rician_paths(
        fc=float(params["fc"]),
        velocity_mps=float(params["velocity_mps"]),
        num_paths=int(params["num_paths"]),
        k_factor_db=float(params.get("k_factor_db", 0.0)),
        delay_spread=float(params["delay_spread"]),
        seed=seed,
    )


def simulate_channel_responses(
    params: dict[str, float | str],
    num_time_samples: int = 512,
    num_frequency_samples: int = 512,
    num_space_samples: int = 512,
    time_step: float = 1e-4,
    seed: int | None = None,
    add_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clean channel responses for one independent realization."""
    channel = _sample_channel(params, seed=seed)
    h_time = synthesize_time_response(channel, num_time_samples, time_step)
    h_freq = synthesize_frequency_response(channel, float(params["bandwidth"]), num_frequency_samples)
    h_space = synthesize_spatial_response(channel, float(params["antenna_spacing"]), num_space_samples)

    if add_noise:
        snr_db = float(params.get("snr_db", 30.0))
        h_time = add_awgn(h_time, snr_db, None if seed is None else seed + 101)
        h_freq = add_awgn(h_freq, snr_db, None if seed is None else seed + 211)
        h_space = add_awgn(h_space, snr_db, None if seed is None else seed + 307)
    return h_time, h_space, h_freq


def _single_curve_from_response(response: np.ndarray, curve_points: int) -> np.ndarray:
    numerator, power = correlation_sufficient_statistics(response, curve_points, remove_mean=False)
    return finalize_correlation_magnitude(numerator, power)


def simulate_curves(
    params: dict[str, float | str],
    curve_points: int = 128,
    include_ccf: bool = False,
    num_time_samples: int = 512,
    num_frequency_samples: int = 512,
    num_space_samples: int = 512,
    time_step: float = 1e-4,
    seed: int | None = None,
    add_noise: bool = False,
) -> np.ndarray:
    """Return one-realization `|R|` curves for plotting/smoke checks."""
    h_time, h_space, h_freq = simulate_channel_responses(
        params,
        num_time_samples=num_time_samples,
        num_frequency_samples=num_frequency_samples,
        num_space_samples=num_space_samples,
        time_step=time_step,
        seed=seed,
        add_noise=add_noise,
    )
    acf = _single_curve_from_response(h_time, curve_points)
    fcf = _single_curve_from_response(h_freq, curve_points)
    if include_ccf:
        ccf = _single_curve_from_response(h_space, curve_points)
        return np.concatenate([acf, ccf, fcf]).astype(np.float32)
    return np.concatenate([acf, fcf]).astype(np.float32)


def simulate_statistical_curves(
    params: dict[str, float | str],
    curve_points: int = 128,
    include_ccf: bool = False,
    num_realizations: int = 16,
    seed: int | None = None,
    num_time_samples: int = 512,
    num_frequency_samples: int = 512,
    num_space_samples: int = 512,
    time_step: float = 1e-4,
) -> np.ndarray:
    """Estimate ensemble `|R|` by averaging complex statistics before magnitude."""
    if num_realizations < 1:
        raise ValueError("num_realizations must be >= 1.")

    acf_num_sum = np.zeros(curve_points, dtype=np.complex128)
    ccf_num_sum = np.zeros(curve_points, dtype=np.complex128)
    fcf_num_sum = np.zeros(curve_points, dtype=np.complex128)
    acf_power_sum = 0.0
    ccf_power_sum = 0.0
    fcf_power_sum = 0.0

    base_seed = 0 if seed is None else int(seed)
    for idx in range(num_realizations):
        h_time, h_space, h_freq = simulate_channel_responses(
            params,
            num_time_samples=num_time_samples,
            num_frequency_samples=num_frequency_samples,
            num_space_samples=num_space_samples,
            time_step=time_step,
            seed=base_seed + 1009 * idx,
            add_noise=False,
        )
        acf_num, acf_power = correlation_sufficient_statistics(h_time, curve_points, remove_mean=False)
        fcf_num, fcf_power = correlation_sufficient_statistics(h_freq, curve_points, remove_mean=False)
        acf_num_sum += acf_num
        fcf_num_sum += fcf_num
        acf_power_sum += acf_power
        fcf_power_sum += fcf_power
        if include_ccf:
            ccf_num, ccf_power = correlation_sufficient_statistics(h_space, curve_points, remove_mean=False)
            ccf_num_sum += ccf_num
            ccf_power_sum += ccf_power

    acf = finalize_correlation_magnitude(acf_num_sum, acf_power_sum)
    fcf = finalize_correlation_magnitude(fcf_num_sum, fcf_power_sum)
    if include_ccf:
        ccf = finalize_correlation_magnitude(ccf_num_sum, ccf_power_sum)
        return np.concatenate([acf, ccf, fcf]).astype(np.float32)
    return np.concatenate([acf, fcf]).astype(np.float32)


def curve_coordinates_for_params(
    params: dict[str, float | str],
    curve_points: int,
    num_frequency_samples: int = 512,
    time_step: float = 1e-4,
) -> dict[str, np.ndarray]:
    lag_indices = np.arange(curve_points, dtype=np.int64)
    return {
        "lag_indices": lag_indices,
        "time_lags_s": lag_indices.astype(np.float64) * time_step,
        "spatial_distances_lambda": lag_indices.astype(np.float64) * float(params["antenna_spacing"]),
        "frequency_offsets_hz": lag_indices.astype(np.float64)
        * (float(params["bandwidth"]) / max(num_frequency_samples - 1, 1)),
    }


def generate_dataset(
    samples: int,
    curve_points: int = 128,
    include_ccf: bool = False,
    seed: int = 42,
    overrides: dict[str, float | str] | None = None,
    num_realizations: int = 16,
    num_time_samples: int = 512,
    num_frequency_samples: int = 512,
    num_space_samples: int = 512,
    time_step: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    rng = np.random.default_rng(seed)
    x_rows = []
    y_rows = []
    raw_params = []
    for index in range(samples):
        params = sample_params(rng, overrides)
        curves = simulate_statistical_curves(
            params,
            curve_points=curve_points,
            include_ccf=include_ccf,
            num_realizations=num_realizations,
            seed=seed + index * 13,
            num_time_samples=num_time_samples,
            num_frequency_samples=num_frequency_samples,
            num_space_samples=num_space_samples,
            time_step=time_step,
        )
        x_rows.append(params_to_vector(params))
        y_rows.append(curves)
        raw_params.append(params)
    return np.stack(x_rows), np.stack(y_rows), raw_params


def split_indices(
    samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(samples)
    train_end = int(samples * train_ratio)
    val_end = train_end + int(samples * val_ratio)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _split_coordinate_arrays(
    raw_params: list[dict[str, float | str]],
    indices: np.ndarray,
    curve_points: int,
    num_frequency_samples: int,
    time_step: float,
) -> dict[str, np.ndarray]:
    lag_indices = np.arange(curve_points, dtype=np.int64)
    spatial = []
    frequency = []
    for idx in indices:
        coords = curve_coordinates_for_params(raw_params[int(idx)], curve_points, num_frequency_samples, time_step)
        spatial.append(coords["spatial_distances_lambda"])
        frequency.append(coords["frequency_offsets_hz"])
    return {
        "lag_indices": lag_indices,
        "time_lags_s": lag_indices.astype(np.float64) * time_step,
        "spatial_distances_lambda": np.stack(spatial).astype(np.float64),
        "frequency_offsets_hz": np.stack(frequency).astype(np.float64),
    }


def save_splits(
    x: np.ndarray,
    y: np.ndarray,
    raw_params: list[dict[str, float | str]],
    output_dir: str | Path,
    curve_points: int,
    include_ccf: bool,
    num_realizations: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_time_samples: int = 512,
    num_frequency_samples: int = 512,
    num_space_samples: int = 512,
    time_step: float = 1e-4,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = split_indices(len(x), train_ratio, val_ratio, seed)
    target_names = ["acf", "ccf", "fcf"] if include_ccf else ["acf", "fcf"]
    for split_name, indices in splits.items():
        params_json = np.array([json.dumps(raw_params[int(i)]) for i in indices])
        coords = _split_coordinate_arrays(raw_params, indices, curve_points, num_frequency_samples, time_step)
        np.savez_compressed(
            output / f"{split_name}.npz",
            x=x[indices],
            y=y[indices],
            params_json=params_json,
            split_indices=indices.astype(np.int64),
            curve_points=np.array([curve_points], dtype=np.int64),
            include_ccf=np.array([include_ccf], dtype=np.bool_),
            num_realizations=np.array([num_realizations], dtype=np.int64),
            num_time_samples=np.array([num_time_samples], dtype=np.int64),
            num_space_samples=np.array([num_space_samples], dtype=np.int64),
            num_frequency_samples=np.array([num_frequency_samples], dtype=np.int64),
            time_step=np.array([time_step], dtype=np.float64),
            target_names=np.array(target_names),
            feature_names=np.array(FEATURE_NAMES),
            label_definition=np.array(["normalized_ensemble_correlation_magnitude"]),
            **coords,
        )


def save_dataset_metadata(
    output_dir: str | Path,
    samples: int,
    curve_points: int,
    include_ccf: bool,
    num_realizations: int,
    seed: int,
    num_time_samples: int = 512,
    num_frequency_samples: int = 512,
    num_space_samples: int = 512,
    time_step: float = 1e-4,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": samples,
        "curve_points": curve_points,
        "include_ccf": include_ccf,
        "num_realizations": num_realizations,
        "seed": seed,
        "num_time_samples": num_time_samples,
        "num_space_samples": num_space_samples,
        "num_frequency_samples": num_frequency_samples,
        "time_step": time_step,
        "label_definition": "normalized ensemble correlation magnitude |E[h h*] / E[|h|^2]|; magnitude is taken after ensemble aggregation",
        "feature_names": FEATURE_NAMES,
        "coordinate_formulas": {
            "lag_indices": "0..curve_points-1",
            "time_lags_s": "lag_indices * time_step",
            "spatial_distances_lambda": "lag_indices * sample_antenna_spacing",
            "frequency_offsets_hz": "lag_indices * sample_bandwidth / (num_frequency_samples - 1)",
        },
        "notes": "Formal labels are clean channel statistics without AWGN/SNR. Rayleigh and Rician are sampled through explicit channel_type/is_rician.",
    }
    (output / "formal_dataset_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
