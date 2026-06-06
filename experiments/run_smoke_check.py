from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channel_sim.correlation import compute_frequency_fcf, compute_temporal_acf
from channel_sim.rayleigh import generate_rayleigh_channel, sample_rayleigh_paths, synthesize_frequency_response


def main() -> None:
    h = generate_rayleigh_channel(fc=3.5e9, v=60 / 3.6, num_paths=8, seed=7)
    channel = sample_rayleigh_paths(fc=3.5e9, velocity_mps=60 / 3.6, num_paths=8, delay_spread=100e-9, seed=7)
    H = synthesize_frequency_response(channel, bandwidth=20e6)
    acf = compute_temporal_acf(h, 64)
    fcf = compute_frequency_fcf(H, 64)
    print(f"h shape: {h.shape}")
    print(f"ACF shape: {acf.shape}, first={acf[0]:.3f}, min={acf.min():.3f}, max={acf.max():.3f}")
    print(f"FCF shape: {fcf.shape}, first={fcf[0]:.3f}, min={fcf.min():.3f}, max={fcf.max():.3f}")


if __name__ == "__main__":
    main()
