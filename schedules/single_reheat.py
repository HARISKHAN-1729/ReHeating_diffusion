"""
Family 2: Single reheat (bump) schedule.
One interruption at a chosen point — the minimal non-monotonic schedule.
Answers: "Does any reheating help at all?"
"""

import numpy as np
from .monotonic import _cosine_alpha_bar


def make(
    N: int = 100,
    t_reheat: float = 0.4,
    delta: float = 0.15,
    **kwargs,
) -> np.ndarray:
    """
    Single reheat schedule.

    At fraction t_reheat through the forward process, adds a Gaussian bump
    of magnitude delta to alpha_bar, creating a region where alpha_bar
    temporarily increases (less noise) then decreases (reheat) before resuming.

    Args:
        N: Number of schedule steps.
        t_reheat: Position of reheat as fraction of total steps [0.2, 0.8].
        delta: Magnitude of the bump [0.05, 0.5].

    Returns:
        alpha_bar array of length N in sampling order.
    """
    t = np.linspace(0, 1, N)
    base = _cosine_alpha_bar(t)

    idx = int(N * t_reheat)
    width = max(int(N * 0.08), 3)
    bump = delta * np.exp(-0.5 * ((np.arange(N) - idx) / (width / 2)) ** 2)
    fwd = np.clip(base + bump, 0.001, 0.999)

    fwd[0] = 0.9999
    fwd[-1] = 0.0001
    return fwd[::-1].copy()
