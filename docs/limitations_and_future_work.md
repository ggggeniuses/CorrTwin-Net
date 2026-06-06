# Limitations and Future Work

## Limitations

1. The current simulator is lightweight and is not equivalent to standardized 3GPP, GBSM, BDCM, or ray-tracing channel models.
2. The dataset is generated from simulation rather than real measurement campaigns.
3. The surrogate model is designed for fast correlation-function prediction during parameter sweeps; it does not directly replace CSI estimation or receiver algorithms.
4. OOD generalization only evaluates selected parameter shifts and does not imply full real-world robustness.
5. Current CCF/ACF/FCF theory validation is a sanity check, not a complete proof of physical fidelity under all channel assumptions.

## Future Work

- Connect the label generator to RIS-V2V GBSM/BDCM simulations.
- Add Sionna, DeepMIMO, or ray-tracing generated data.
- Add uncertainty estimation to detect unreliable OOD predictions.
- Add physics-informed regularization to improve curve smoothness and physical consistency.
- Add larger ablations over dataset size, curve length, model capacity, and noise level.
