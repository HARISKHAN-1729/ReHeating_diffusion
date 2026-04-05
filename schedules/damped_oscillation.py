"""
Family 4: Damped oscillation schedule.
Large oscillations early (global structure), smoothly transitioning to monotonic.
Formula: alpha_bar(t) = cosine(t) + A * exp(-gamma*t) * sin(2*pi*f*t)
"""

import numpy as np
from .monotonic import _cosine_alpha_bar


def make(
    N: int = 100,
    A: float = 0.2,
    gamma: float = 2.5,
    freq: float = 4.0,
    **kwargs,
) -> np.ndarray:
    """
    Damped oscillation schedule.

    Adds a damped sinusoidal perturbation to the cosine base schedule.
    Large oscillations at early timesteps (high noise), decaying toward
    monotonic at late timesteps (low noise / fine details).

    Args:
        N: Number of schedule steps.
        A: Amplitude of oscillation.
        gamma: Damping rate.
        freq: Oscillation frequency.

    Returns:
        alpha_bar array of length N in sampling order.
    """
    t = np.linspace(0, 1, N)
    base = _cosine_alpha_bar(t)

    pert = A * np.exp(-gamma * t) * np.sin(2 * np.pi * freq * t)
    fwd = np.clip(base + pert, 0.001, 0.999)

    fwd[0] = 0.9999
    fwd[-1] = 0.0001
    return fwd[::-1].copy()
