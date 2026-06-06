from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CorrTwin-Net reproduction pipeline.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--formal", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Pipeline step failed: {' '.join(cmd)}") from exc


def main() -> None:
    args = parse_args()
    py = sys.executable
    if args.quick:
        samples, curve_points, epochs, ood_samples = "120", "64", "3", "120"
        rf_estimators = "20"
        realizations = "16"
        theory_realizations = "256"
        data_dir = "data"
        results_dir = "results"
    else:
        samples, curve_points, epochs, ood_samples = "10000", "128", "100", "2000"
        rf_estimators = "100"
        realizations = "100"
        theory_realizations = "2000"
        data_dir = "data_formal_p0"
        results_dir = "results_formal_p0"

    run([py, "experiments/run_theory_validation.py", "--curve-points", curve_points, "--num-realizations", theory_realizations, "--validate-actual-generator", "--results-dir", results_dir])
    run([py, "experiments/run_generate_dataset.py", "--samples", samples, "--curve-points", curve_points, "--include-ccf", "--num-realizations", realizations, "--output-dir", data_dir])
    run([py, "experiments/run_train_mlp.py", "--epochs", epochs, "--batch-size", "64", "--data-dir", data_dir, "--results-dir", results_dir])
    run([py, "experiments/run_train_resmlp.py", "--epochs", epochs, "--batch-size", "64", "--data-dir", data_dir, "--results-dir", results_dir])
    run([py, "experiments/run_train_sklearn_baselines.py", "--rf-estimators", rf_estimators, "--data-dir", data_dir, "--results-dir", results_dir])
    run([py, "experiments/run_evaluate_curves.py", "--model", "mlp", "--data-dir", data_dir, "--results-dir", results_dir])
    run([py, "experiments/run_evaluate_curves.py", "--model", "resmlp", "--data-dir", data_dir, "--results-dir", results_dir])
    run([py, "experiments/run_compare_all_models.py", "--results-dir", results_dir])
    run([py, "experiments/run_speed_comparison.py", "--model", "resmlp", "--runs", "50" if args.quick else "300", "--num-realizations", realizations, "--results-dir", results_dir])
    run([py, "experiments/run_ood_generalization.py", "--all", "--samples", ood_samples, "--curve-points", curve_points, "--include-ccf", "--num-realizations", realizations, "--results-dir", results_dir])
    if args.formal:
        run([py, "experiments/run_realization_convergence.py", "--curve-points", curve_points, "--reference-realizations", theory_realizations, "--results-dir", results_dir])
        run([py, "scripts/build_artifacts_manifest.py", "--data-dir", data_dir, "--results-dir", results_dir, "--output", "artifacts_manifest_formal_p0.generated.json"])
        run([py, "scripts/verify_artifacts.py", "--manifest", "artifacts_manifest_formal_p0.generated.json"])

    print("Generated key outputs:")
    for pattern in [f"{results_dir}/figures/*.png", f"{results_dir}/metrics/*.csv", f"{results_dir}/metrics/*.json"]:
        for path in sorted(ROOT.glob(pattern)):
            print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
