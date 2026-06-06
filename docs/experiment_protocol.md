# Experiment Protocol

## Data Generation

The simulator samples Rayleigh/Rician multipath channels under the following scenario parameters:

- Channel type: Rayleigh or Rician.
- Carrier frequency: 2.4, 3.5, 5.9, or 28 GHz.
- Velocity: 0 to 120 km/h.
- Number of paths: 4, 8, 12, or 16.
- Rician K-factor: 0 to 15 dB for Rician samples; Rayleigh samples use `is_rician=0`.
- Antenna spacing: lambda/8, lambda/4, or lambda/2.
- Delay spread: 50 ns to 500 ns.
- Bandwidth: 5, 10, or 20 MHz.

## Input Features

Each input vector contains normalized values:

```text
[is_rician, fc, velocity, num_paths, k_factor_db, antenna_spacing, delay_spread, bandwidth]
```

The explicit `is_rician` feature prevents Rayleigh and Rician-K-near-zero cases from being ambiguous.

## Output Labels

Formal experiments use:

```text
y = concat([|ACF|_N, |CCF|_N, |FCF|_N])
```

Quick smoke tests may use:

```text
y = concat([|ACF|_N, |FCF|_N])
```

## Correlation Functions

- Temporal ACF is computed from time-domain channel response autocorrelation.
- Spatial CCF is computed from spatial response autocorrelation.
- Frequency FCF is computed from frequency response autocorrelation.
- The prediction target is normalized ensemble correlation magnitude, not signed or complex correlation.
- Formal labels are clean channel statistics: AWGN is not added and SNR is not a formal input feature.
- Complex sufficient statistics are averaged across independent channel realizations before magnitude is taken.
- Curves are normalized to `[0, 1]`.

## Models

- MLP: feed-forward PyTorch baseline.
- ResMLP: residual PyTorch model.
- Mean Curve: predicts the global mean training curve.
- Channel-Type Mean: predicts separate mean curves for Rayleigh and Rician samples.
- Ridge: linear multi-output regression.
- RandomForest: nonlinear tree baseline.
- KNN: local interpolation baseline.

## Metrics

The project reports:

- Overall MSE, MAE, relative error, and curve correlation.
- Per-curve ACF/CCF/FCF MSE, MAE, and correlation.
- Per-sample inference time.

## Theory Validation

The theory script compares Monte Carlo correlation magnitudes with:

- Temporal `|ACF|`: Clarke/Jakes `|J0(2*pi*fD*tau)|`.
- Spatial `|CCF|`: isotropic Bessel reference.
- Frequency `|FCF|`: exponential power-delay-profile reference.

## OOD Scenarios

OOD tests include:

- `velocity_ood`
- `k_factor_ood`
- `path_num_ood`
- `delay_spread_ood`
- `bandwidth_ood`

Each scenario generates a separate synthetic test set with shifted parameters.

## Speed Test

The speed comparison measures average seconds per sample for:

- Traditional simulator label generation.
- Neural forward inference in single-sample mode.
- Neural forward inference in batched-throughput mode.

## Reproduction Commands

```bash
python experiments/run_full_pipeline.py --quick
python experiments/run_full_pipeline.py --formal
```
