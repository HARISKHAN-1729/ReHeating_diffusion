"""
Family 3: Periodic sawtooth schedule.
Monotonic decrease with periodic partial re-noising.
Answers: "Does repeated reheating compound benefits?"
"""

import numpy as np
from .monotonic import _cosine_alpha_bar


def make(
    N: int = 100,
    period: int = 25,
    sawtooth_delta: float = 0.08,
    **kwargs,
) -> np.ndarray:
    """
    Periodic sawtooth schedule.

    Every `period` steps in the forward process, adds a partial re-noising
    bump of magnitude sawtooth_delta.

    Args:
        N: Number of schedule steps.
        period: Steps between each sawtooth bump.
        sawtooth_delta: Magnitude of each bump.

    Returns:
        alpha_bar array of length N in sampling order.
    """
    t = np.linspace(0, 1, N)
    base = _cosine_alpha_bar(t)
    fwd = base.copy()

    for i in range(period, N - 5, period):
        w = min(5, N - i)
        for j in range(w):
            idx = min(i + j, N - 1)
            fwd[idx] = min(fwd[idx] + sawtooth_delta * (1 - j / w) * fwd[i], 0.999)

    fwd[0] = 0.9999
    fwd[-1] = 0.0001
    return fwd[::-1].copy()
