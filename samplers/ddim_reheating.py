"""
Generalized DDIM sampler supporting non-monotonic alpha_bar schedules.
Identical to standard DDIM when schedule is monotonic.
Also includes the adaptive reheating sampler.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class AdaptiveStats:
    """Diagnostics from adaptive sampling."""
    reheat_positions: List[int]
    prediction_deltas: List[float]
    total_nfe: int
    total_reheats: int


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
    use_fp16: bool = True,
) -> Union[Tensor, Tuple[Tensor, list]]:
    """
    Generalized DDIM sampler supporting non-monotonic alpha_bar schedules.

    Args:
        model: UNet that takes (x_t, timestep_int) -> eps_pred
        alpha_bars: from make_schedule(), sampling order (index 0=noise, N-1=clean)
        shape: (batch_size, channels, height, width)
        device: compute device
        eta: 0=deterministic DDIM, 1=stochastic DDPM-like
        clip_denoised: whether to clip x0 predictions to [-1, 1]
        num_train_timesteps: model's training timestep count (typically 1000)
        return_trajectory: whether to return intermediate states
        use_fp16: use float16 for model forward passes (saves VRAM)

    Returns:
        Generated samples (or tuple of samples and trajectory list).
    """
    ab = torch.tensor(alpha_bars, dtype=torch.float32, device=device)
    x = torch.randn(shape, device=device)
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(ab) - 1):
        a_c, a_n = ab[i], ab[i + 1]

        # Convert alpha_bar to integer timestep for model input
        ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
        ts = ts.expand(shape[0])

        # Model forward pass
        if use_fp16 and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps = model(x, ts)
            eps = eps.float()
        else:
            eps = model(x, ts)

        # Predict x0
        x0 = (x - (1 - a_c).sqrt() * eps) / a_c.sqrt()
        if clip_denoised:
            x0 = x0.clamp(-1, 1)

        # Recompute eps from clipped x0 for consistency
        eps = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()

        # DDIM/DDPM step — works for both denoising and reheating
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

    if return_trajectory:
        return x0, traj
    return x0


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
    use_fp16: bool = True,
) -> Union[Tensor, Tuple[Tensor, AdaptiveStats]]:
    """
    Adaptive reheating sampler.
    Triggers reheat when prediction change exceeds threshold tau.
    Respects NFE budget: total NFE <= len(base_alpha_bars) + max_reheats.
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

        if use_fp16 and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps = model(x, ts)
            eps = eps.float()
        else:
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
                    and i < len(schedule) - 5):
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
