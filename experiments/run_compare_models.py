from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plotting import plot_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved MLP and ResMLP metrics.")
    parser.add_argument("--metric", default="mse", choices=["mse", "mae", "relative_error", "curve_correlation"])
    return parser.parse_args()


def load_metrics(model: str) -> dict[str, float]:
    path = ROOT / "results" / "metrics" / f"{model}_eval_metrics.json"
    if not path.exists():
        path = ROOT / "results" / "metrics" / f"{model}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = []
    for model in ["mlp", "resmlp"]:
        metrics = load_metrics(model)
        row = {"model": model}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "metrics" / "model_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    plot_bar(
        df["model"].tolist(),
        df[args.metric].astype(float).tolist(),
        f"Model Comparison ({args.metric})",
        args.metric,
        ROOT / "results" / "figures" / "model_comparison.png",
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
