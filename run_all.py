from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable
    run([py, "experiments/run_smoke_check.py"])
    run([py, "experiments/run_plot_example_curves.py", "--curve-points", "64"])
    run([py, "experiments/run_generate_dataset.py", "--samples", "300", "--curve-points", "64"])
    run([py, "experiments/run_train_mlp.py", "--epochs", "10"])
    run([py, "experiments/run_train_resmlp.py", "--epochs", "10"])
    run([py, "experiments/run_evaluate_curves.py", "--model", "mlp"])
    run([py, "experiments/run_evaluate_curves.py", "--model", "resmlp"])
    run([py, "experiments/run_compare_models.py"])
    run([py, "experiments/run_generalization_test.py", "--model", "mlp", "--samples", "50"])
    run([py, "experiments/run_speed_comparison.py", "--model", "mlp", "--runs", "50"])


if __name__ == "__main__":
    main()
