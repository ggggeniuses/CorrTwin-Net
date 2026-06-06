"""Simplified Rayleigh multipath channel generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .doppler import path_doppler_shifts, wavelength


@dataclass(frozen=True)
class MultipathChannel:
    gains: np.ndarray
    angles_rad: np.ndarray
    delays_s: np.ndarray
    dopplers_hz: np.ndarray
    fc: float


def sample_rayleigh_paths(
    fc: float,
    velocity_mps: float,
    num_paths: int,
    delay_spread: float,
    seed: int | None = None,
) -> MultipathChannel:
    """Sample a normalized Rayleigh multipath realization."""
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1.")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=num_paths)
    gains = (
        rng.normal(size=num_paths) + 1j * rng.normal(size=num_paths)
    ) / np.sqrt(2.0 * num_paths)
    delays = rng.exponential(scale=max(delay_spread, 1e-12), size=num_paths)
    dopplers = path_doppler_shifts(fc, velocity_mps, angles)
    return MultipathChannel(gains, angles, delays, dopplers, fc)


def synthesize_time_response(
    channel: MultipathChannel,
    num_samples: int = 512,
    time_step: float = 1e-4,
) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float64) * time_step
    phase = np.exp(1j * 2.0 * np.pi * channel.dopplers_hz[:, None] * t[None, :])
    return np.sum(channel.gains[:, None] * phase, axis=0)


def synthesize_frequency_response(
    channel: MultipathChannel,
    bandwidth: float,
    num_samples: int = 512,
) -> np.ndarray:
    freqs = np.linspace(-bandwidth / 2.0, bandwidth / 2.0, num_samples)
    phase = np.exp(-1j * 2.0 * np.pi * channel.delays_s[:, None] * freqs[None, :])
    return np.sum(channel.gains[:, None] * phase, axis=0)


def synthesize_spatial_response(
    channel: MultipathChannel,
    antenna_spacing: float = 0.5,
    num_samples: int = 512,
) -> np.ndarray:
    lam = wavelength(channel.fc)
    positions = np.arange(num_samples, dtype=np.float64) * antenna_spacing * lam
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        * positions[None, :]
        * np.sin(channel.angles_rad[:, None])
        / lam
    )
    return np.sum(channel.gains[:, None] * phase, axis=0)


def add_awgn(signal: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = (
        rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    ) * np.sqrt(noise_power / 2.0)
    return signal + noise


def generate_rayleigh_channel(
    fc: float,
    v: float,
    num_paths: int,
    delay_spread: float = 100e-9,
    num_samples: int = 512,
    time_step: float = 1e-4,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a time-domain Rayleigh channel response.

    This keeps the smoke-check API from the project plan:
    `h = generate_rayleigh_channel(fc=3.5e9, v=60/3.6, num_paths=8)`.
    """
    channel = sample_rayleigh_paths(fc, v, num_paths, delay_spread, seed)
    return synthesize_time_response(channel, num_samples, time_step)
