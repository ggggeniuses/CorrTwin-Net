# CorrTwin-Net Project Report

## Goal

CorrTwin-Net predicts clean-channel normalized ensemble correlation magnitudes:

```text
scenario parameters -> normalized ensemble |Temporal ACF| / |Spatial CCF| / |Frequency FCF|
```

It is a surrogate modeling project, not CSI estimation and not a neural receiver.

## Corrected Task Definition

For each scenario, the project estimates conditional ensemble correlation functions by aggregating complex sufficient statistics over multiple independent channel realizations:

```text
R[k] = sum_i mean_n(h_i[n] conj(h_i[n+k])) / sum_i mean_n(|h_i[n]|^2)
target[k] = |R[k]|
```

The implementation computes `|E[R]|`, not `E[|R|]`. The formal task uses clean channel responses. AWGN is not added and SNR is not a model input.

## Formal Results

The completed formal benchmark uses 10000 samples, 128 curve points, and 100 independent realizations per label.

| Model | Overall MSE | Overall MAE | Overall Corr |
|---|---:|---:|---:|
| ResMLP | 0.001383 | 0.022161 | 0.972651 |
| RandomForest | 0.001472 | 0.024120 | 0.959782 |
| MLP | 0.001898 | 0.025610 | 0.962286 |
| KNN | 0.004949 | 0.040444 | 0.946940 |
| Ridge | 0.010081 | 0.066477 | 0.815992 |
| Channel-Type Mean | 0.024805 | 0.113154 | 0.847950 |
| Mean Curve | 0.105983 | 0.290361 | 0.838050 |

Actual-generator theory validation:

```text
ACF MSE = 2.50e-07
CCF MSE = 8.52e-06
FCF MSE = 3.24e-05
max_abs_error = 0.00879
```

Speed comparison:

```text
single-realization simulator = 0.002486 s/sample
100-realization ensemble simulator = 0.238618 s/sample
ResMLP single-sample latency = 0.000737 s/sample
ResMLP batched throughput = 0.000005 s/sample
```

Realization convergence with an `N=2000` reference gives `N=100` label MSE about `1.76e-4`.

## Boundary

The simulator is still lightweight and does not replace 3GPP, GBSM, BDCM, ray tracing, or measured channel datasets. The current target is normalized ensemble correlation magnitude, so the model does not predict full signed or complex correlation functions. The OOD evaluation covers finite parameter shifts only and does not prove universal real-world generalization. The project is best understood as a theory-checked surrogate modeling prototype for clean-channel parameter sweeps.
