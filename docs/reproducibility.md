# Reproducibility

## Environment

Recommended Python version:

```text
Python 3.10 or 3.11
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For an exact local reference, see `requirements-lock.txt`.

## Formal Reproduction

The current completed formal run uses:

```text
samples = 10000
curve_points = 128
epochs = 100
target = normalized ensemble |ACF| + |CCF| + |FCF|
num_realizations = 100
OOD samples = 2000 per scenario
data_dir = data_formal_p0
results_dir = results_formal_p0
```

Reproduce with:

```bash
python experiments/run_theory_validation.py --curve-points 128 --num-realizations 2000 --validate-actual-generator --results-dir results_formal_p0
python experiments/run_generate_dataset.py --samples 10000 --curve-points 128 --include-ccf --num-realizations 100 --output-dir data_formal_p0
python experiments/run_train_mlp.py --epochs 100 --batch-size 64 --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_train_resmlp.py --epochs 100 --batch-size 64 --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_train_sklearn_baselines.py --rf-estimators 100 --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_evaluate_curves.py --model mlp --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_evaluate_curves.py --model resmlp --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_compare_all_models.py --results-dir results_formal_p0
python experiments/run_ood_generalization.py --all --samples 2000 --curve-points 128 --include-ccf --num-realizations 100 --results-dir results_formal_p0
python experiments/run_speed_comparison.py --model resmlp --runs 300 --num-realizations 100 --results-dir results_formal_p0
python experiments/run_realization_convergence.py --curve-points 128 --reference-realizations 2000 --results-dir results_formal_p0
python scripts/build_artifacts_manifest.py --data-dir data_formal_p0 --results-dir results_formal_p0 --output artifacts_manifest_formal_p0.generated.json
python scripts/verify_artifacts.py --manifest artifacts_manifest_formal_p0.generated.json
```

## Quick Check

Use quick mode only to verify mechanics:

```bash
python experiments/run_full_pipeline.py --quick
```

Quick mode is not the reported formal result.

## Tests

```bash
python -m pytest -q
```

The latest local run passed 22 tests. The pytest cache warning in this environment is only a cache-write permission warning.

## Notes

- GPU acceleration is used automatically when `torch.cuda.is_available()` is true and `--device auto` is used.
- Formal labels are clean channel statistics and do not include AWGN/SNR.
- The task predicts normalized ensemble correlation magnitudes, not full signed or complex correlations.
