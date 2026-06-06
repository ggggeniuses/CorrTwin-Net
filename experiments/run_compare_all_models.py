from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sklearn, MLP, and ResMLP metrics.")
    parser.add_argument("--metric", default="overall_mse", choices=["overall_mse", "overall_mae", "overall_corr"])
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding training/evaluation script first.")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_metrics(model: str, metrics: dict) -> dict:
    return {
        "model": model,
        "overall_mse": metrics.get("overall_mse", metrics.get("mse")),
        "overall_mae": metrics.get("overall_mae", metrics.get("mae")),
        "overall_corr": metrics.get("overall_corr", metrics.get("curve_correlation")),
        "acf_mse": metrics.get("acf_mse"),
        "ccf_mse": metrics.get("ccf_mse"),
        "fcf_mse": metrics.get("fcf_mse"),
    }


def main() -> None:
    args = parse_args()
    results_root = ROOT / args.results_dir
    metrics_dir = results_root / "metrics"
    rows = []
    for model in ["mlp", "resmlp"]:
        rows.append(normalize_metrics(model, load_json(metrics_dir / f"{model}_eval_metrics.json")))
    sklearn = load_json(metrics_dir / "sklearn_baselines.json")
    for model, metrics in sklearn.items():
        rows.append(normalize_metrics(model, metrics))

    df = pd.DataFrame(rows)
    out_csv = metrics_dir / "all_model_comparison.csv"
    df.to_csv(out_csv, index=False)

    fig_path = results_root / "figures" / "all_model_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df[args.metric])
    plt.ylabel(args.metric)
    plt.title("All Model Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
