"""
Family 4: Damped oscillation schedule.
A sinusoidal perturbation on top of the linear descent in timestep space.
Large oscillations early (where global structure is decided), smoothly
decaying to monotonic at the end (where fine details are set).
"""

import numpy as np


def make(
    N: int = 100,
    num_train_timesteps: int = 1000,
    A: float = 0.2,
    gamma: float = 2.5,
    freq: float = 4.0,
    **kwargs,
) -> np.ndarray:
    """
    Damped oscillation schedule in timestep space.

    timestep(i) = base(i) + A * T * exp(-gamma * t) * sin(2*pi*freq*t)

    where base(i) is a linear ramp from T-1 to 0, and t = i/N in [0, 1].

    Args:
        N: number of sampling steps.
        num_train_timesteps: model's T.
        A: oscillation amplitude (fraction of T).
        gamma: damping rate.
        freq: oscillation frequency (number of full cycles).

    Returns:
        Integer timestep array of length N+1 in sampling order.
    """
    T = num_train_timesteps
    t_norm = np.linspace(0, 1, N + 1)

    # Base: linear descent from T-1 to 0
    base = (1 - t_norm) * (T - 1)

    # Damped sinusoidal perturbation
    perturbation = A * (T - 1) * np.exp(-gamma * t_norm) * np.sin(2 * np.pi * freq * t_norm)

    timesteps = np.clip(base + perturbation, 0, T - 1).round().astype(int)
    timesteps[0] = T - 1
    timesteps[-1] = 0
    return timesteps
