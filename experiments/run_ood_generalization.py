from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.dataset_generator import generate_dataset
from experiments.common import device_from_arg, load_checkpoint
from experiments.run_evaluate_curves import build_model
from models.sklearn_baselines import load_sklearn_model
from utils.config import load_yaml_config
from utils.io_utils import save_json
from utils.metrics import segmented_curve_metrics
from utils.normalization import FEATURE_NAMES
from utils.plotting import plot_curve_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD generalization tests.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--curve-points", type=int, default=128)
    parser.add_argument("--include-ccf", action="store_true")
    parser.add_argument("--num-realizations", type=int, default=16)
    parser.add_argument("--config", type=str, default="configs/ood_generalization.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def load_models(device: torch.device, results_root: Path) -> dict:
    models = {}
    for name in ["mlp", "resmlp"]:
        ckpt_path = results_root / "checkpoints" / f"{name}_best.pt"
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, device)
            model = build_model(ckpt).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            models[name] = ("torch", model)

    sklearn_dir = results_root / "runs" / "sklearn_baselines"
    for name in ["ridge", "random_forest", "knn"]:
        path = sklearn_dir / f"{name}.pkl"
        if path.exists():
            models[name] = ("sklearn", load_sklearn_model(path))
    if not models:
        raise FileNotFoundError("No trained models found. Train MLP/ResMLP and sklearn baselines first.")
    return models


def coerce_overrides(overrides: dict) -> dict:
    fixed = {}
    for key, value in overrides.items():
        if key == "channel_type":
            fixed[key] = str(value)
        elif key == "num_paths":
            fixed[key] = int(value)
        else:
            fixed[key] = float(value)
    return fixed


def audit_ood_dataset(scenario: str, raw_params: list[dict], x: np.ndarray) -> dict:
    report: dict[str, object] = {
        "samples": len(raw_params),
        "feature_names": FEATURE_NAMES,
        "normalized_feature_min": np.min(x, axis=0).astype(float).tolist(),
        "normalized_feature_max": np.max(x, axis=0).astype(float).tolist(),
    }
    if scenario == "velocity_ood":
        values = np.array([float(p["velocity_mps"]) for p in raw_params])
        report["all_outside_training"] = bool(np.all(values > 120.0 / 3.6))
        report["velocity_mps_min"] = float(values.min())
        report["velocity_mps_max"] = float(values.max())
    elif scenario == "k_factor_ood":
        types = [str(p["channel_type"]) for p in raw_params]
        values = np.array([float(p["k_factor_db"]) for p in raw_params])
        report["all_rician"] = bool(all(t == "rician" for t in types))
        report["all_k_factor_18_db"] = bool(np.allclose(values, 18.0))
    elif scenario == "path_num_ood":
        values = np.array([int(p["num_paths"]) for p in raw_params])
        report["all_num_paths_24"] = bool(np.all(values == 24))
    elif scenario == "delay_spread_ood":
        values = np.array([float(p["delay_spread"]) for p in raw_params])
        report["all_outside_training"] = bool(np.all(values > 500e-9))
    elif scenario == "bandwidth_ood":
        values = np.array([float(p["bandwidth"]) for p in raw_params])
        report["all_outside_training"] = bool(np.all(values > 20e6))
    return report


def predict(kind: str, model, x, device: torch.device):
    if kind == "torch":
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()
    return model.predict(x)


def plot_ood_bar(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index="scenario", columns="model", values="overall_mse")
    pivot.plot(kind="bar", figsize=(10, 4))
    plt.ylabel("overall_mse")
    plt.title("OOD Generalization")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    results_root = ROOT / args.results_dir
    config = load_yaml_config(ROOT / args.config)
    scenarios = config.get("scenarios", {})
    if args.all:
        selected = scenarios
    else:
        scenario = args.scenario or "velocity_ood"
        if scenario not in scenarios:
            raise KeyError(f"Unknown scenario {scenario}. Available: {list(scenarios)}")
        selected = {scenario: scenarios[scenario]}

    device = device_from_arg(args.device)
    models = load_models(device, results_root)
    rows = []
    audit = {}
    first_example = None
    for scenario_index, (scenario, overrides) in enumerate(selected.items()):
        overrides = coerce_overrides(overrides)
        x, y, raw_params = generate_dataset(
            samples=args.samples,
            curve_points=args.curve_points,
            include_ccf=args.include_ccf,
            num_realizations=args.num_realizations,
            seed=1000 + 997 * scenario_index,
            overrides=overrides,
        )
        audit[scenario] = audit_ood_dataset(scenario, raw_params, x)
        failed_flags = [key for key, value in audit[scenario].items() if key.startswith("all_") and value is not True]
        if failed_flags:
            raise RuntimeError(f"OOD audit failed for {scenario}: {failed_flags}")
        for model_name, (kind, model) in models.items():
            pred = predict(kind, model, x, device)
            if pred.shape[1] != y.shape[1]:
                print(f"skip {model_name} for {scenario}: output dim {pred.shape[1]} != target dim {y.shape[1]}")
                continue
            metrics = segmented_curve_metrics(pred, y, args.curve_points, args.include_ccf)
            rows.append({"scenario": scenario, "model": model_name, **metrics})
            if first_example is None and model_name == "mlp":
                first_example = (scenario, y[0], pred[0])

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No OOD rows produced. Check that trained model output dimensions match this dataset.")
    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "ood_generalization.csv", index=False)
    save_json(metrics_dir / "ood_generalization.json", {"rows": rows})
    save_json(metrics_dir / "ood_dataset_audit.json", audit)
    plot_ood_bar(df, results_root / "figures" / "ood_generalization_bar.png")

    if first_example:
        scenario, true, pred = first_example
        cp = args.curve_points
        plot_curve_pair(true[:cp], pred[:cp], f"OOD ACF Example ({scenario})", results_root / "figures" / "ood_prediction_examples.png")
    print(df[["scenario", "model", "overall_mse", "overall_mae", "overall_corr"]].to_string(index=False))


if __name__ == "__main__":
    main()
