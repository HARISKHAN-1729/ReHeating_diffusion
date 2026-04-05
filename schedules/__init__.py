"""
Schedule construction for reheating diffusion sampling.
Five families: monotonic, single_reheat, sawtooth, damped_oscillation, adaptive.
"""

from .monotonic import make as make_monotonic, make_linear, make_edm_sigma
from .single_reheat import make as make_single_reheat
from .sawtooth import make as make_sawtooth
from .damped_oscillation import make as make_damped_oscillation
from .adaptive import make as make_adaptive_base, calibrate_threshold

# Unified entry point
from .monotonic import _cosine_alpha_bar

import numpy as np
from typing import Optional


def make_schedule(
    family: str,
    N: int = 100,
    **kwargs,
) -> np.ndarray:
    """
    Unified schedule construction.

    Returns alpha_bar array of length N in sampling order:
      index 0 = high noise (low alpha_bar), index N-1 = low noise (high alpha_bar ~1).

    For non-monotonic schedules, some consecutive entries have alpha_bar[i+1] < alpha_bar[i]
    (reheat steps).
    """
    makers = {
        'monotonic': make_monotonic,
        'monotonic_ddpm': make_monotonic,
        'linear': make_linear,
        'single_reheat': make_single_reheat,
        'sawtooth': make_sawtooth,
        'damped_oscillation': make_damped_oscillation,
    }
    if family not in makers:
        raise ValueError(f"Unknown schedule family: {family}. Options: {list(makers.keys())}")
    return makers[family](N=N, **kwargs)


def make_edm2_schedule(
    family: str,
    N: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    **kwargs,
) -> np.ndarray:
    """
    Create sigma schedule for EDM2-style sampling.
    Returns sigma array ordered for sampling: high sigma -> low sigma,
    with non-monotonic segments where sigma increases (reheat).
    Appends terminal sigma=0.
    """
    alpha_bars = make_schedule(family, N=N, **kwargs)
    sigmas = np.sqrt((1 - alpha_bars) / (alpha_bars + 1e-10))
    sigmas = np.clip(sigmas, sigma_min, sigma_max)
    sigmas = np.append(sigmas, 0.0)
    return sigmas


def make_flow_schedule(
    family: str,
    N: int = 50,
    **kwargs,
) -> np.ndarray:
    """
    Create time schedule for flow matching.
    Returns t values ordered for sampling: t near 0 (noise) -> t near 1 (data).
    Non-monotonic segments have t decreasing (backward steps).
    """
    alpha_bars = make_schedule(family, N=N, **kwargs)
    times = alpha_bars * 0.998 + 0.001  # Map [0,1] -> [0.001, 0.999]
    return times


def schedule_report(alpha_bars: np.ndarray, name: str = '') -> dict:
    """Compute and print statistics for a schedule."""
    diffs = np.diff(alpha_bars)
    n_reheat = int(np.sum(diffs < 0))
    n_denoise = int(np.sum(diffs > 0))

    snr = alpha_bars / (1 - alpha_bars + 1e-10)
    log_snr = np.log(snr + 1e-10)
    diffs_snr = np.diff(log_snr)

    path = float(np.sum(np.abs(diffs_snr)))
    net = float(log_snr[-1] - log_snr[0])
    overhead = path / abs(net) - 1 if abs(net) > 1e-6 else 0

    log10_snr = np.log10(snr + 1e-10)
    in_critical = np.sum(np.abs(log10_snr) < 0.5)

    stats = {
        'name': name,
        'N': len(alpha_bars),
        'reheat_steps': n_reheat,
        'denoise_steps': n_denoise,
        'overhead': overhead,
        'critical_zone_steps': int(in_critical),
    }

    if name:
        print(f"--- {name} ---")
        for k, v in stats.items():
            if k != 'name':
                print(f"  {k}: {v}")
    return stats


def compare_all_schedules(N: int = 50):
    """Print comparison of all schedule families."""
    families = {
        'Monotonic (DDIM)': ('monotonic', {}),
        'Single Reheat': ('single_reheat', {'t_reheat': 0.4, 'delta': 0.15}),
        'Sawtooth': ('sawtooth', {'period': 15, 'sawtooth_delta': 0.08}),
        'Damped Oscillation': ('damped_oscillation', {'A': 0.2, 'gamma': 2.5, 'freq': 4}),
    }

    print(f"\n{'Schedule':<30} {'Steps':>5} {'Reheat':>7} {'CritZone':>9} {'Overhead':>9}")
    print("-" * 65)
    for name, (fam, kw) in families.items():
        sched = make_schedule(fam, N=N, **kw)
        stats = schedule_report(sched)
        print(f"{name:<30} {stats['N']:>5} {stats['reheat_steps']:>7} "
              f"{stats['critical_zone_steps']:>9} {stats['overhead']:>8.1%}")
