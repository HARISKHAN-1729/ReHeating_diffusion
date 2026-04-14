"""
Family 3: Periodic sawtooth schedule.
Every `period` steps, jump backward to a higher timestep (reheat),
then resume denoising. Tests whether repeated reheating compounds benefits.
"""

import numpy as np
from . import monotonic


def make(
    N: int = 100,
    num_train_timesteps: int = 1000,
    period: int = 25,
    sawtooth_delta: float = 0.08,
    **kwargs,
) -> np.ndarray:
    """
    Sawtooth schedule: periodic backward jumps in timestep space.

    Args:
        N: number of sampling steps.
        num_train_timesteps: model's T.
        period: steps between each reheat.
        sawtooth_delta: fraction of current timestep to jump back.

    Returns:
        Integer timestep array of length N+1 in sampling order.
    """
    base = monotonic.make(N=N, num_train_timesteps=num_train_timesteps)
    result = base.copy()

    for i in range(period, N - 2, period):
        current_ts = int(result[i])
        if current_ts < 5:
            continue  # Don't reheat near the end
        jump = max(int(current_ts * sawtooth_delta), 1)
        result[i] = min(current_ts + jump, num_train_timesteps - 1)

    result[0] = num_train_timesteps - 1
    result[-1] = 0
    return result
