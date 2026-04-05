"""
Family 1: Monotonic baseline schedules.
Cosine, linear, and EDM sigma schedules — the control conditions.
"""

import numpy as np


def _cosine_alpha_bar(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return f / f[0]


def make(N: int = 100, **kwargs) -> np.ndarray:
    """
    Monotonic cosine schedule.

    Returns alpha_bar array of length N in sampling order:
      index 0 = high noise (low alpha_bar), index N-1 = low noise (high alpha_bar).
    """
    t = np.linspace(0, 1, N)
    fwd = _cosine_alpha_bar(t)
    fwd[0] = 0.9999
    fwd[-1] = 0.0001
    return fwd[::-1].copy()


def make_linear(N: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02, **kwargs) -> np.ndarray:
    """
    Linear beta schedule (original DDPM).
    Returns alpha_bar in sampling order.
    """
    betas = np.linspace(beta_start, beta_end, N)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    alpha_bars[0] = max(alpha_bars[0], 0.9999)
    alpha_bars[-1] = min(alpha_bars[-1], 0.0001)
    return alpha_bars[::-1].copy()


def make_edm_sigma(N: int = 50, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, **kwargs) -> np.ndarray:
    """
    EDM sigma schedule (Karras et al., 2022).
    Returns sigma array (not alpha_bar) in sampling order: high sigma -> low sigma.
    Appends terminal sigma=0.
    """
    step_indices = np.arange(N)
    sigmas = (sigma_max ** (1 / rho) + step_indices / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = np.append(sigmas, 0.0)
    return sigmas
