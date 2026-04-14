"""
Family 2: Single reheat (bump) schedule.
One interruption: at a chosen point, jump BACK to a higher timestep (more noise),
then resume denoising. The minimal non-monotonic schedule.
"""

import numpy as np
from . import monotonic


def make(
    N: int = 100,
    num_train_timesteps: int = 1000,
    t_reheat: float = 0.4,
    delta: float = 0.15,
    **kwargs,
) -> np.ndarray:
    """
    Single reheat schedule.

    At fraction t_reheat through the sampling process, jumps backward by
    delta * current_timestep, then resumes monotonic denoising.

    Args:
        N: number of sampling steps.
        num_train_timesteps: model's T (typically 1000).
        t_reheat: when to reheat, as fraction of sampling progress [0.2, 0.8].
                  0.4 means "40% of the way from noise to clean".
        delta: how far to jump back, as fraction of current timestep [0.05, 0.5].

    Returns:
        Integer timestep array of length N+1 in sampling order.
    """
    base = monotonic.make(N=N, num_train_timesteps=num_train_timesteps)

    # Reheat insertion point (in sampling order: index 0 is highest timestep)
    reheat_idx = int(N * t_reheat)
    reheat_idx = max(2, min(reheat_idx, N - 3))

    # Current timestep at the reheat point
    current_ts = int(base[reheat_idx])
    # Jump back: increase timestep by delta fraction
    jump = max(int(current_ts * delta), 1)
    reheat_ts = min(current_ts + jump, num_train_timesteps - 1)

    # Build the schedule:
    #   [base[:reheat_idx], reheat_ts, linear from reheat_ts to 0]
    prefix = list(base[:reheat_idx])
    prefix.append(reheat_ts)  # The backward jump

    # Resume: linear from reheat_ts down to 0
    remaining = N + 1 - len(prefix)
    if remaining > 0:
        suffix = np.linspace(reheat_ts, 0, remaining + 1).round().astype(int)
        prefix.extend(suffix[1:].tolist())  # skip first (already appended)

    result = np.array(prefix[:N + 1], dtype=int)
    result[0] = num_train_timesteps - 1
    result[-1] = 0
    return result
