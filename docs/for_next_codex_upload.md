# Handoff Guide for the Next Codex

This project is a development preview after the 2026-06-06 full整改 pass. Do not describe it as a final v1.0 release until the long formal run and realization convergence study are completed.

## Current Local Paths

```text
project = D:\communication\CorrTwin-Net
current_data = data_formal_p0
current_results = results_formal_p0
artifact_manifest = artifacts_manifest_formal_p0.generated.json
```

## Correct Task Definition

The formal target is:

```text
scenario parameters -> normalized ensemble |ACF| / |CCF| / |FCF|
```

Important corrections:

- Formal labels are clean channel statistics; AWGN is not added.
- `snr_db` is removed from the formal input vector.
- Complex correlation sufficient statistics are ensemble-averaged before taking magnitude.
- Rayleigh PDP sampling no longer double-applies exponential delay weighting.
- Rician total power is normalized and LOS/NLOS ratio follows K.
- OOD generation writes an audit JSON.
- Speed comparison uses N-realization ensemble simulation as the main simulator baseline.

## Current Verified Benchmark

```text
samples = 10000
curve_points = 128
num_realizations = 100
epochs = 100
OOD samples = 2000 per scenario
```

Main result:

```text
ResMLP        overall_mse=0.001383, overall_corr=0.972651
RandomForest  overall_mse=0.001472, overall_corr=0.959782
MLP           overall_mse=0.001898, overall_corr=0.962286
KNN           overall_mse=0.004949, overall_corr=0.946940
Ridge         overall_mse=0.010081, overall_corr=0.815992
```

Actual-generator theory validation:

```text
ACF MSE = 2.50e-07
CCF MSE = 8.52e-06
FCF MSE = 3.24e-05
max_abs_error = 0.00879
```

Speed:

```text
single-realization simulator = 0.002486 s/sample
100-realization ensemble simulator = 0.238618 s/sample
ResMLP single-sample latency = 0.000737 s/sample
ResMLP batched throughput = 0.000005 s/sample
```

## Verified Commands

```bash
python -m pytest -q
python experiments/run_theory_validation.py --curve-points 128 --num-realizations 2000 --validate-actual-generator --results-dir results_formal_p0
python experiments/run_generate_dataset.py --samples 10000 --curve-points 128 --include-ccf --num-realizations 100 --output-dir data_formal_p0
python experiments/run_train_mlp.py --epochs 100 --batch-size 64 --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_train_resmlp.py --epochs 100 --batch-size 64 --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_train_sklearn_baselines.py --rf-estimators 100 --data-dir data_formal_p0 --results-dir results_formal_p0
python experiments/run_compare_all_models.py --results-dir results_formal_p0
python experiments/run_ood_generalization.py --all --samples 2000 --curve-points 128 --include-ccf --num-realizations 100 --results-dir results_formal_p0
python experiments/run_speed_comparison.py --model resmlp --runs 300 --num-realizations 100 --results-dir results_formal_p0
python experiments/run_realization_convergence.py --curve-points 128 --reference-realizations 2000 --results-dir results_formal_p0
python scripts/build_artifacts_manifest.py --data-dir data_formal_p0 --results-dir results_formal_p0 --output artifacts_manifest_formal_p0.generated.json
python scripts/verify_artifacts.py --manifest artifacts_manifest_formal_p0.generated.json
```

## Upload Guidance

If uploading to GitHub, commit source code, tests, configs, docs, and lightweight results. Large `.npz`, `.pt`, and `.pkl` artifacts can be included in the prepared local zip or uploaded through GitHub Releases instead of normal Git history.
