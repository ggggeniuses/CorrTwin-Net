# Implementation Status

## Current Status

CorrTwin-Net is currently a `development preview`, not a final v1.0 release. The latest整改版本 implements the clean-channel task requested by the负责人方案:

```text
scenario parameters -> normalized ensemble |ACF| / |CCF| / |FCF|
```

The formal label is computed by accumulating complex correlation sufficient statistics across independent realizations and taking the magnitude after ensemble aggregation.

## Completed Locally

- Removed `snr_db` from the formal input feature vector.
- Disabled AWGN for formal channel-statistics labels.
- Fixed Rayleigh exponential PDP sampling: delays follow an exponential distribution and gains are not double-weighted by delay.
- Fixed Rician power normalization so total power is 1 and LOS/NLOS power ratio matches K.
- Implemented complex sufficient-statistics correlation aggregation.
- Added per-sample physical coordinates for time, space, and frequency lags.
- Added actual-generator theory validation in addition to analytic Monte Carlo validation.
- Added OOD dataset audit output.
- Fixed speed comparison to use full N-realization ensemble simulation as the main baseline.
- Added artifact hash manifest generation and verification scripts.
- Added physical tests for ensemble aggregation, coordinates, OOD sampling, Rician power, PDP/FCF consistency, generator theory sanity, and best checkpoint evaluation.
- Added realization convergence experiment.

## Verified Formal Benchmark

```text
samples = 10000
curve_points = 128
num_realizations = 100 per label
target = normalized ensemble |ACF| + |CCF| + |FCF|
epochs = 100
data_dir = data_formal_p0
results_dir = results_formal_p0
```

| Model | Overall MSE | Overall MAE | Overall Corr | ACF MSE | CCF MSE | FCF MSE |
|---|---:|---:|---:|---:|---:|---:|
| ResMLP | 0.001383 | 0.022161 | 0.972651 | 0.002942 | 0.000575 | 0.000630 |
| RandomForest | 0.001472 | 0.024120 | 0.959782 | 0.002523 | 0.000946 | 0.000948 |
| MLP | 0.001898 | 0.025610 | 0.962286 | 0.004411 | 0.000616 | 0.000666 |
| KNN | 0.004949 | 0.040444 | 0.946940 | 0.011333 | 0.001392 | 0.002123 |
| Ridge | 0.010081 | 0.066477 | 0.815992 | 0.019052 | 0.001629 | 0.009563 |

## Theory Validation

Actual generator validation with 2000 realizations:

```text
ACF MSE = 2.50e-07
CCF MSE = 8.52e-06
FCF MSE = 3.24e-05
max_abs_error = 0.00879
```

These satisfy the整改方案 threshold: each MSE < 0.01 and max absolute error < 0.15.

## Speed Result

| Method | Seconds per sample |
|---|---:|
| Single-realization simulator | 0.002486 |
| 100-realization ensemble simulator | 0.238618 |
| ResMLP single-sample latency | 0.000737 |
| ResMLP batched throughput | 0.000005 |

The main speedup should be reported against the 100-realization ensemble simulator.

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
