from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import load_npz_dataset
from models.sklearn_baselines import (
    save_sklearn_model,
    train_knn_baseline,
    train_random_forest_baseline,
    train_ridge_baseline,
)
from utils.io_utils import create_run_dir, save_json, save_yaml
from utils.metrics import segmented_curve_metrics
from utils.seed import set_global_seed


def predict_global_mean(y_train, n_rows: int):
    return y_train.mean(axis=0, keepdims=True).repeat(n_rows, axis=0)


def predict_channel_type_mean(x_train, y_train, x_test):
    global_mean = y_train.mean(axis=0)
    pred = []
    for row in x_test:
        is_rician = row[0] >= 0.5
        mask = x_train[:, 0] >= 0.5 if is_rician else x_train[:, 0] < 0.5
        pred.append(y_train[mask].mean(axis=0) if mask.any() else global_mean)
    return pd.DataFrame(pred).to_numpy(dtype=y_train.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn baselines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=80)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--knn-neighbors", type=int, default=7)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    arrays, labels, curve_points, include_ccf = load_npz_dataset(args.data_dir)
    results_root = ROOT / args.results_dir
    run_dir = create_run_dir("sklearn_baselines", results_root / "runs", args.run_name)
    model_dir = results_root / "runs" / "sklearn_baselines"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", vars(args) | {"curve_points": curve_points, "include_ccf": include_ccf})

    trainers = {
        "ridge": lambda: train_ridge_baseline(arrays["train"], labels["train"]),
        "random_forest": lambda: train_random_forest_baseline(
            arrays["train"],
            labels["train"],
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
        ),
        "knn": lambda: train_knn_baseline(arrays["train"], labels["train"], n_neighbors=args.knn_neighbors),
    }

    results = {}
    rows = []
    simple_predictions = {
        "mean_curve": predict_global_mean(labels["train"], len(labels["test"])),
        "channel_type_mean": predict_channel_type_mean(arrays["train"], labels["train"], arrays["test"]),
    }
    for name, pred in simple_predictions.items():
        metrics = segmented_curve_metrics(pred, labels["test"], curve_points, include_ccf)
        results[name] = metrics
        rows.append({"model": name, **metrics})

    for name, train_fn in trainers.items():
        print(f"training {name}...")
        model = train_fn()
        pred = model.predict(arrays["test"])
        metrics = segmented_curve_metrics(pred, labels["test"], curve_points, include_ccf)
        results[name] = metrics
        rows.append({"model": name, **metrics})
        save_sklearn_model(model, model_dir / f"{name}.pkl")
        save_sklearn_model(model, run_dir / f"{name}.pkl")

    metrics_dir = results_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / "sklearn_baselines.json", results)
    save_json(run_dir / "metrics.json", results)
    pd.DataFrame(rows).to_csv(metrics_dir / "sklearn_baselines.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "sklearn_baselines.csv", index=False)
    print(pd.DataFrame(rows)[["model", "overall_mse", "overall_mae", "overall_corr"]].to_string(index=False))


if __name__ == "__main__":
    main()
