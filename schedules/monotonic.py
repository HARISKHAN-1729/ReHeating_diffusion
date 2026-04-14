"""
Family 1: Monotonic baseline schedule.
Evenly-spaced integer timesteps from T-1 down to 0.
"""

import numpy as np


def make(N: int = 100, num_train_timesteps: int = 1000, **kwargs) -> np.ndarray:
    """
    Monotonic schedule: evenly spaced timesteps from T-1 to 0.

    Returns:
        Integer timestep array of length N+1 (includes terminal 0).
        Ordered for sampling: [999, ..., 0].
    """
    timesteps = np.linspace(num_train_timesteps - 1, 0, N + 1).round().astype(int)
    timesteps[0] = num_train_timesteps - 1
    timesteps[-1] = 0
    return timesteps
