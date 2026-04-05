"""
Reheating Diffusion Sampler v2
==============================
Generalized sampler supporting non-monotonic noise schedules.
Works with DDPM, EDM2, and Flow Matching — no retraining required.

Key changes from v1:
  - Adaptive reheating (Family 5) as first-class citizen
  - EDM2 sigma-based parameterization support
  - Flow matching backward-step support
  - DDPM stochastic baseline properly included
  - Calibration utility for adaptive threshold

Usage:
    # DDPM
    schedule = make_schedule('damped_oscillation', N=50)
    images = ddim_sample(model, schedule, shape=(64,3,32,32))

    # Adaptive
    tau = calibrate_threshold(model, shape=(16,3,32,32), percentile=80)
    images = adaptive_sample(model, make_schedule('monotonic', N=50),
                             shape=(64,3,32,32), tau=tau)

    # EDM2
    sigmas = make_edm2_schedule('damped_oscillation', N=50)
    images = edm2_sample(model, sigmas, shape=(64,3,32,32))
"""

import numpy as np
import torch
from torch import Tensor
from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# Schedule Construction
# =============================================================================

def _cosine_alpha_bar(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return f / f[0]


def make_schedule(
    family: Literal[
        'monotonic', 'single_reheat', 'sawtooth',
        'damped_oscillation', 'monotonic_ddpm'
    ],
    N: int = 100,
    t_reheat: float = 0.4,
    delta: float = 0.15,
    period: int = 25,
    sawtooth_delta: float = 0.08,
    A: float = 0.2,
    gamma: float = 2.5,
    freq: float = 4.0,
) -> np.ndarray:
    """
    Create alpha_bar schedule for sampling.

    Returns array of length N, ordered for sampling:
      index 0 → high noise (low alpha), index N-1 → low noise (high alpha).
    Non-monotonic schedules have segments where alpha DECREASES (reheat).
    """
    t = np.linspace(0, 1, N)
    base = _cosine_alpha_bar(t)  # 1 → ~0

    if family in ('monotonic', 'monotonic_ddpm'):
        fwd = base
    elif family == 'single_reheat':
        fwd = base.copy()
        idx = int(N * t_reheat)
        width = max(int(N * 0.08), 3)
        bump = delta * np.exp(-0.5 * ((np.arange(N) - idx) / (width / 2)) ** 2)
        fwd = np.clip(fwd + bump, 0.001, 0.999)
    elif family == 'sawtooth':
        fwd = base.copy()
        for i in range(period, N - 5, period):
            w = min(5, N - i)
            for j in range(w):
                idx = min(i + j, N - 1)
                fwd[idx] = min(fwd[idx] + sawtooth_delta * (1 - j / w) * fwd[i], 0.999)
    elif family == 'damped_oscillation':
        pert = A * np.exp(-gamma * t) * np.sin(2 * np.pi * freq * t)
        fwd = np.clip(base + pert, 0.001, 0.999)
    else:
        raise ValueError(f"Unknown family: {family}")

    fwd[0] = 0.9999
    fwd[-1] = 0.0001
    return fwd[::-1].copy()  # Reverse for sampling order


def make_edm2_schedule(
    family: str, N: int = 50, sigma_min: float = 0.002, sigma_max: float = 80.0,
    **kwargs
) -> np.ndarray:
    """
    Create sigma schedule for EDM2-style sampling.
    Returns sigma array ordered for sampling: high sigma → low sigma,
    with non-monotonic segments where sigma increases (reheat).
    """
    alpha_bars = make_schedule(family, N=N, **kwargs)
    # Convert alpha_bar to sigma: sigma = sqrt((1-alpha)/alpha)
    sigmas = np.sqrt((1 - alpha_bars) / (alpha_bars + 1e-10))
    sigmas = np.clip(sigmas, sigma_min, sigma_max)
    sigmas = np.append(sigmas, 0.0)  # Terminal sigma=0
    return sigmas


def make_flow_schedule(
    family: str, N: int = 50, **kwargs
) -> np.ndarray:
    """
    Create time schedule for flow matching.
    Returns t values ordered for sampling: t near 0 (noise) → t near 1 (data).
    Non-monotonic segments have t decreasing (backward steps).
    """
    alpha_bars = make_schedule(family, N=N, **kwargs)
    # In flow matching, t=1 is data, t=0 is noise
    # alpha_bar ≈ t² approximately, so t ≈ sqrt(alpha_bar) is reasonable
    # But simpler: just linearly map alpha_bar to [eps, 1-eps]
    times = alpha_bars * 0.998 + 0.001  # Map [0,1] → [0.001, 0.999]
    return times


# =============================================================================
# Sampler 1: Generalized DDIM (for DDPM-trained models)
# =============================================================================

@torch.no_grad()
def ddim_sample(
    model,
    alpha_bars: np.ndarray,
    shape: tuple,
    device: str = 'cuda',
    eta: float = 0.0,
    clip_denoised: bool = True,
    num_train_timesteps: int = 1000,
    return_trajectory: bool = False,
) -> Tensor:
    """
    Generalized DDIM sampler supporting non-monotonic alpha_bar schedules.
    Identical to standard DDIM when schedule is monotonic.

    Args:
        model: UNet that takes (x_t, timestep_int) → eps_pred
        alpha_bars: from make_schedule(), sampling order
        eta: 0=deterministic DDIM, 1=stochastic DDPM-like
    """
    ab = torch.tensor(alpha_bars, dtype=torch.float32, device=device)
    x = torch.randn(shape, device=device)
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(ab) - 1):
        a_c, a_n = ab[i], ab[i + 1]

        # Convert alpha_bar to integer timestep for model input
        ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        ts = ts.expand(shape[0])

        eps = model(x, ts)
        x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)

        # Recompute eps from clipped x0
        eps = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()

        # DDIM/DDPM step — works for both directions
        is_denoising = a_n > a_c
        if eta > 0 and is_denoising:
            var = (1 - a_n) / (1 - a_c) * (1 - a_c / a_n)
            sigma = eta * var.clamp(min=0).sqrt()
            direction = (1 - a_n - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_n.sqrt() * x0 + direction + sigma * torch.randn_like(x)
        else:
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps

        if return_trajectory:
            traj.append(x.cpu().clone())

    return (x0, traj) if return_trajectory else x0


# =============================================================================
# Sampler 2: Adaptive Reheating
# =============================================================================

@dataclass
class AdaptiveStats:
    """Diagnostics from adaptive sampling."""
    reheat_positions: List[int]     # Which steps triggered reheat
    prediction_deltas: List[float]  # Δ at each step
    total_nfe: int
    total_reheats: int


@torch.no_grad()
def calibrate_threshold(
    model,
    shape: tuple,
    alpha_bars: Optional[np.ndarray] = None,
    n_calibration: int = 100,
    percentile: float = 80.0,
    device: str = 'cuda',
    num_train_timesteps: int = 1000,
) -> float:
    """
    Calibrate the adaptive reheat threshold τ.
    Runs n_calibration monotonic samples and computes the percentile
    of prediction-change magnitudes.

    Takes ~2 minutes for 100 calibration samples.
    """
    if alpha_bars is None:
        alpha_bars = make_schedule('monotonic', N=50)

    ab = torch.tensor(alpha_bars, dtype=torch.float32, device=device)
    all_deltas = []

    for _ in range(n_calibration):
        x = torch.randn((1, *shape[1:]), device=device)
        prev_x0 = None

        for i in range(len(ab) - 1):
            a_c = ab[i]
            ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
            ts = ts.unsqueeze(0)

            eps = model(x, ts)
            x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)

            if prev_x0 is not None:
                delta = (x0 - prev_x0).pow(2).mean().sqrt().item()
                all_deltas.append(delta)

            prev_x0 = x0.clone()

            a_n = ab[i + 1]
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()

    tau = float(np.percentile(all_deltas, percentile))
    print(f"Calibrated τ = {tau:.4f} (percentile {percentile}, "
          f"from {len(all_deltas)} measurements)")
    return tau


@torch.no_grad()
def adaptive_sample(
    model,
    base_alpha_bars: np.ndarray,
    shape: tuple,
    tau: float,
    delta: float = 0.15,
    device: str = 'cuda',
    eta: float = 0.0,
    num_train_timesteps: int = 1000,
    max_reheats: int = 15,
    return_stats: bool = False,
) -> Tensor:
    """
    Adaptive reheating sampler.
    Triggers reheat when prediction change exceeds threshold τ.
    Respects NFE budget: total NFE ≤ len(base_alpha_bars) + max_reheats.
    """
    schedule = list(base_alpha_bars)
    device_t = torch.device(device)
    x = torch.randn(shape, device=device_t)
    prev_x0 = None
    nfe = 0
    n_reheats = 0
    max_nfe = len(base_alpha_bars) + max_reheats
    stats = AdaptiveStats([], [], 0, 0) if return_stats else None

    i = 0
    while i < len(schedule) - 1 and nfe < max_nfe:
        a_c = torch.tensor(schedule[i], dtype=torch.float32, device=device_t)
        ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        ts = ts.expand(shape[0])

        eps = model(x, ts)
        x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)
        eps = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()
        nfe += 1

        # Check prediction stability
        if prev_x0 is not None:
            delta_pred = (x0 - prev_x0).pow(2).mean().sqrt().item()
            if return_stats:
                stats.prediction_deltas.append(delta_pred)

            # Trigger reheat if unstable and budget allows
            if (delta_pred > tau
                    and n_reheats < max_reheats
                    and i < len(schedule) - 5):  # Keep budget for final steps
                a_reheat = float(a_c.item()) * (1 - delta)
                a_reheat = max(a_reheat, 0.001)
                schedule.insert(i + 1, a_reheat)
                n_reheats += 1
                if return_stats:
                    stats.reheat_positions.append(i)

        prev_x0 = x0.clone()

        # Step to next noise level
        a_n = torch.tensor(schedule[i + 1], dtype=torch.float32, device=device_t)
        is_denoising = a_n > a_c
        if eta > 0 and is_denoising:
            var = (1 - a_n) / (1 - a_c) * (1 - a_c / a_n)
            sigma = eta * var.clamp(min=0).sqrt()
            direction = (1 - a_n - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_n.sqrt() * x0 + direction + sigma * torch.randn_like(x)
        else:
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps

        i += 1

    if return_stats:
        stats.total_nfe = nfe
        stats.total_reheats = n_reheats
        return x0, stats
    return x0


# =============================================================================
# Sampler 3: EDM2 with Reheating
# =============================================================================

@torch.no_grad()
def edm2_sample(
    denoiser,
    sigmas: np.ndarray,
    shape: tuple,
    device: str = 'cuda',
    S_churn: float = 0.0,
    S_noise: float = 1.0,
    return_trajectory: bool = False,
) -> Tensor:
    """
    EDM2-style Euler sampler with non-monotonic sigma schedule.
    denoiser: D(x; sigma) → denoised x.
    sigmas: from make_edm2_schedule(), includes terminal 0.

    The key: d = (x - D(x,sigma)) / sigma is the update direction.
    x_next = x + (sigma_next - sigma_cur) * d
    When sigma_next > sigma_cur, this is a reheat step — it adds noise.
    """
    sig = torch.tensor(sigmas, dtype=torch.float32, device=device)
    x = torch.randn(shape, device=device) * sig[0]
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(sig) - 1):
        s_cur = sig[i]
        s_next = sig[i + 1]

        if s_cur == 0:
            break

        # Evaluate denoiser
        d_cur = denoiser(x, s_cur.expand(shape[0]))
        # EDM2 direction: (x - D(x,sigma)) / sigma
        d = (x - d_cur) / s_cur

        # Euler step (works for both directions)
        x = x + (s_next - s_cur) * d

        if return_trajectory:
            traj.append(x.cpu().clone())

    return (d_cur, traj) if return_trajectory else d_cur


# =============================================================================
# Sampler 4: Flow Matching with Reheating
# =============================================================================

@torch.no_grad()
def flow_sample(
    velocity_model,
    times: np.ndarray,
    shape: tuple,
    device: str = 'cuda',
    return_trajectory: bool = False,
) -> Tensor:
    """
    Flow matching Euler sampler with non-monotonic time schedule.
    velocity_model: v(x, t) → velocity field
    times: from make_flow_schedule(), ordered noise→data (low t → high t)

    Backward steps (t decreasing) use negative dt — valid Euler step.
    """
    t_arr = torch.tensor(times, dtype=torch.float32, device=device)
    x = torch.randn(shape, device=device)
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(t_arr) - 1):
        t_cur = t_arr[i]
        t_next = t_arr[i + 1]
        dt = t_next - t_cur  # Can be negative (reheat)

        v = velocity_model(x, t_cur.expand(shape[0]))
        x = x + dt * v

        if return_trajectory:
            traj.append(x.cpu().clone())

    return (x, traj) if return_trajectory else x


# =============================================================================
# Analysis Utilities
# =============================================================================

def schedule_report(alpha_bars: np.ndarray, name: str = '') -> dict:
    """Compute and print statistics for a schedule."""
    diffs = np.diff(alpha_bars)
    n_reheat = int(np.sum(diffs < 0))  # Alpha decreasing = reheat in sampling order
    n_denoise = int(np.sum(diffs > 0))

    snr = alpha_bars / (1 - alpha_bars + 1e-10)
    log_snr = np.log(snr + 1e-10)
    diffs_snr = np.diff(log_snr)

    path = float(np.sum(np.abs(diffs_snr)))
    net = float(log_snr[-1] - log_snr[0])
    overhead = path / abs(net) - 1 if abs(net) > 1e-6 else 0

    # Count time in critical zone (|log10_snr| < 0.5)
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


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    compare_all_schedules(N=50)
    compare_all_schedules(N=100)
