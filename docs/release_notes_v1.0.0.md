# CorrTwin-Net Formal P0 Notes

This document summarizes the completed formal P0 run after the 2026-06-06整改.

## Scope

- Clean-channel normalized ensemble `|ACF|/` `|CCF|` / `|FCF|` label generation.
- No AWGN and no SNR in the formal input.
- Complex sufficient-statistics aggregation before magnitude.
- Fixed exponential PDP and Rician power normalization.
- Actual-generator theory validation.
- OOD dataset audit.
- Ensemble-simulation speed comparison.
- Artifact SHA256 manifest and verification scripts.

## Formal Benchmark

```text
samples = 10000
curve_points = 128
num_realizations = 100
epochs = 100
```

| Model | Overall MSE | Overall MAE | Overall Corr |
|---|---:|---:|---:|
| ResMLP | 0.001383 | 0.022161 | 0.972651 |
| RandomForest | 0.001472 | 0.024120 | 0.959782 |
| MLP | 0.001898 | 0.025610 | 0.962286 |
| KNN | 0.004949 | 0.040444 | 0.946940 |
| Ridge | 0.010081 | 0.066477 | 0.815992 |

## Verification

```text
pytest: 22 passed
syntax check: passed
artifact manifest: verified 31 artifacts
```
