"""
Schedule package — produces INTEGER TIMESTEP sequences for DDPM,
SIGMA sequences for EDM2, and TIME sequences for flow matching.
"""

import numpy as np
from . import monotonic, single_reheat, sawtooth, damped_oscillation, adaptive

_FAMILY_MAP = {
    'monotonic': monotonic,
    'single_reheat': single_reheat,
    'sawtooth': sawtooth,
    'damped_oscillation': damped_oscillation,
    'adaptive': adaptive,
}

# =========================================================================
# DDPM integer timestep schedules
# =========================================================================

def make_schedule(family, N=100, num_train_timesteps=1000, **kwargs):
    if family not in _FAMILY_MAP:
        raise ValueError(f"Unknown family: {family}. Available: {list(_FAMILY_MAP.keys())}")
    return _FAMILY_MAP[family].make(N=N, num_train_timesteps=num_train_timesteps, **kwargs)

def make_adaptive_base(N=100, num_train_timesteps=1000):
    return monotonic.make(N=N, num_train_timesteps=num_train_timesteps)

def count_reheat_steps(timesteps):
    return int(np.sum(np.diff(timesteps.astype(float)) > 0))

# =========================================================================
# EDM2 sigma schedules
# =========================================================================

def make_edm2_schedule(family='monotonic', N=50, sigma_min=0.002, sigma_max=80.0, rho=7.0, **kwargs):
    if family == 'monotonic':
        step_indices = np.arange(N)
        sigmas = (sigma_max ** (1/rho) + step_indices / (N-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        return np.append(sigmas, 0.0)

    elif family == 'single_reheat':
        t_reheat = kwargs.get('t_reheat', 0.4)
        delta = kwargs.get('delta', 0.15)
        base = make_edm2_schedule('monotonic', N=N, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        reheat_idx = max(2, min(int(N * t_reheat), N - 3))
        # Jump BACK: set sigma to a value HIGHER than the previous step
        # Look back by delta fraction of remaining steps
        lookback = max(1, int(N * delta * 0.5))
        source_idx = max(0, reheat_idx - lookback)
        base[reheat_idx] = base[source_idx]  # Jump back to an earlier (higher) sigma
        return base

    elif family == 'damped_oscillation':
        A = kwargs.get('A', 0.2)
        gamma = kwargs.get('gamma', 2.5)
        freq = kwargs.get('freq', 4.0)
        base = make_edm2_schedule('monotonic', N=N, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        t_norm = np.linspace(0, 1, N + 1)
        pert = A * np.exp(-gamma * t_norm) * np.sin(2 * np.pi * freq * t_norm)
        # Perturb in log-sigma space — use full amplitude (not 0.3x)
        log_base = np.log(np.clip(base, 1e-10, None))
        log_perturbed = log_base + pert * np.abs(log_base)
        result = np.exp(np.clip(log_perturbed, np.log(sigma_min), np.log(sigma_max)))
        result[-1] = 0.0
        return result

    elif family == 'sawtooth':
        period = kwargs.get('period', 25)
        sd = kwargs.get('sawtooth_delta', 0.08)
        base = make_edm2_schedule('monotonic', N=N, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        for i in range(period, N - 2, period):
            lookback = max(2, int(sd * N * 0.5))
            source = max(0, i - lookback)
            if base[source] > base[i]:
                base[i] = base[source]
        return base
    else:
        raise ValueError(f"Unknown EDM2 schedule family: {family}")

def count_edm2_reheat_steps(sigmas):
    return int(np.sum(np.diff(sigmas[:-1]) > 0))

# =========================================================================
# Flow matching time schedules
# =========================================================================

def make_flow_schedule(family='monotonic', N=50, t_min=0.001, t_max=0.999, **kwargs):
    if family == 'monotonic':
        return np.linspace(t_min, t_max, N + 1)
    elif family == 'single_reheat':
        t_reheat_pos = kwargs.get('t_reheat', 0.4)
        delta = kwargs.get('delta', 0.15)
        base = np.linspace(t_min, t_max, N + 1)
        idx = max(2, min(int(N * t_reheat_pos), N - 3))
        base[idx] = max(base[idx] - base[idx] * delta, t_min)
        return base
    elif family == 'damped_oscillation':
        A = kwargs.get('A', 0.2)
        gamma = kwargs.get('gamma', 2.5)
        freq = kwargs.get('freq', 4.0)
        base = np.linspace(t_min, t_max, N + 1)
        s = np.linspace(0, 1, N + 1)
        pert = A * np.exp(-gamma * s) * np.sin(2 * np.pi * freq * s)
        result = np.clip(base + pert * (t_max - t_min) * 0.3, t_min, t_max)
        result[0] = t_min
        result[-1] = t_max
        return result
    else:
        raise ValueError(f"Unknown flow schedule family: {family}")

def count_flow_reheat_steps(times):
    return int(np.sum(np.diff(times) < 0))

# =========================================================================
# Utilities
# =========================================================================

def schedule_report(timesteps, name=''):
    diffs = np.diff(timesteps.astype(float))
    if timesteps.max() > 10:
        n_reheat = int(np.sum(diffs > 0))
    else:
        n_reheat = int(np.sum(diffs < 0))
    stats = {'name': name, 'N': len(timesteps), 'reheat_steps': n_reheat,
             'denoise_steps': len(timesteps) - 1 - n_reheat}
    if name:
        print(f"  {name}: {len(timesteps)} steps, {n_reheat} reheat")
    return stats

calibrate_threshold = adaptive.calibrate_threshold
