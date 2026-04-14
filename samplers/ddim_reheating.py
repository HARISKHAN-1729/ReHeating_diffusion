"""
Generalized DDIM sampler supporting non-monotonic noise schedules.

CRITICAL FIX: This sampler takes INTEGER TIMESTEPS and the model's own
alphas_cumprod array. At each step, it looks up:
    alpha_bar = alphas_cumprod[timestep]
This guarantees the alpha_bar in the DDIM formula matches exactly what
the model assumed when predicting epsilon. No schedule mismatch possible.

The old version constructed its own cosine alpha_bar values which didn't
match the model's linear-beta training schedule, producing garbage outputs.
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
    timesteps: np.ndarray,
    alphas_cumprod: torch.Tensor,
    shape: tuple,
    device: str = 'cuda',
    eta: float = 0.0,
    clip_denoised: bool = True,
    return_trajectory: bool = False,
    use_fp16: bool = True,
) -> Union[Tensor, Tuple[Tensor, list]]:
    """
    Generalized DDIM sampler with non-monotonic schedule support.

    Args:
        model: UNet(x_t, timestep_int) -> eps_pred
        timesteps: integer timestep sequence from schedules.make_schedule().
                   Ordered for sampling: index 0 = high timestep (noise),
                   last index = 0 (clean). Non-monotonic segments have
                   timesteps that INCREASE (reheat steps).
        alphas_cumprod: model's own alphas_cumprod (1D tensor, length=T).
                        Access via model.alphas_cumprod after loading.
        shape: (batch_size, C, H, W)
        device: compute device
        eta: 0=deterministic DDIM, 1=stochastic DDPM-like
        clip_denoised: clip x0 predictions to [-1, 1]
        return_trajectory: return list of intermediate states
        use_fp16: use float16 for model forward passes

    Returns:
        Generated samples, or (samples, trajectory_list).
    """
    ts_arr = torch.tensor(timesteps, dtype=torch.long, device=device)
    ac = alphas_cumprod.to(device)  # shape [T], e.g. [1000]
    x = torch.randn(shape, device=device)
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(ts_arr) - 1):
        t_cur = ts_arr[i]
        t_next = ts_arr[i + 1]

        # Look up alpha_bar from the MODEL'S OWN schedule
        a_c = ac[t_cur]
        a_n = ac[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

        # Expand timestep for batch dimension
        t_batch = t_cur.expand(shape[0])

        # Model forward pass
        if use_fp16 and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps = model(x, t_batch)
            eps = eps.float()
        else:
            eps = model(x, t_batch)

        # Predict x0 using the CORRECT alpha_bar
        x0 = (x - (1 - a_c).sqrt() * eps) / a_c.sqrt()
        if clip_denoised:
            x0 = x0.clamp(-1, 1)

        # Recompute eps from (possibly clipped) x0
        eps = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()

        # DDIM step — works for BOTH directions:
        #   a_n > a_c (more signal) = denoising step
        #   a_n < a_c (less signal) = reheat step
        is_denoising = a_n > a_c
        if eta > 0 and is_denoising:
            # Stochastic step (only when denoising)
            var = (1 - a_n) / (1 - a_c) * (1 - a_c / a_n)
            sigma = eta * var.clamp(min=0).sqrt()
            direction = (1 - a_n - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_n.sqrt() * x0 + direction + sigma * torch.randn_like(x)
        else:
            # Deterministic step (always for reheat, optionally for denoise)
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps

        if return_trajectory:
            traj.append(x.cpu().clone())

    if return_trajectory:
        return x0, traj
    return x0


@torch.no_grad()
def adaptive_sample(
    model,
    base_timesteps: np.ndarray,
    alphas_cumprod: torch.Tensor,
    shape: tuple,
    tau: float,
    delta_jump: int = 50,
    device: str = 'cuda',
    eta: float = 0.0,
    max_reheats: int = 15,
    return_stats: bool = False,
    use_fp16: bool = True,
) -> Union[Tensor, Tuple[Tensor, AdaptiveStats]]:
    """
    Adaptive reheating sampler.
    Triggers reheat when prediction change between consecutive steps exceeds tau.

    Args:
        model: UNet(x_t, timestep_int) -> eps_pred
        base_timesteps: integer timestep sequence (monotonic base).
        alphas_cumprod: model's alphas_cumprod.
        shape: (batch_size, C, H, W)
        tau: prediction-change threshold (from calibrate_threshold).
        delta_jump: how many integer timesteps to jump BACK when reheating.
                    E.g. 50 means: if at timestep 200, jump back to 250.
        device: compute device
        eta: stochasticity parameter
        max_reheats: budget cap on number of reheat insertions
        return_stats: return AdaptiveStats diagnostics
        use_fp16: use float16 for model forward

    Returns:
        Generated samples, or (samples, AdaptiveStats).
    """
    schedule = list(base_timesteps.astype(int))
    ac = alphas_cumprod.to(device)
    num_ts = len(ac)
    x = torch.randn(shape, device=device)
    prev_x0 = None
    nfe = 0
    n_reheats = 0
    max_nfe = len(base_timesteps) + max_reheats
    stats = AdaptiveStats([], [], 0, 0) if return_stats else None

    i = 0
    while i < len(schedule) - 1 and nfe < max_nfe:
        t_cur = int(schedule[i])
        a_c = ac[t_cur]
        t_batch = torch.tensor([t_cur], dtype=torch.long, device=device).expand(shape[0])

        if use_fp16 and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps = model(x, t_batch)
            eps = eps.float()
        else:
            eps = model(x, t_batch)

        x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)
        eps = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()
        nfe += 1

        # Check prediction stability
        if prev_x0 is not None:
            delta_pred = (x0 - prev_x0).pow(2).mean().sqrt().item()
            if return_stats:
                stats.prediction_deltas.append(delta_pred)

            # Trigger reheat if prediction is unstable and budget allows
            if (delta_pred > tau
                    and n_reheats < max_reheats
                    and i < len(schedule) - 5):
                reheat_ts = min(t_cur + delta_jump, num_ts - 1)
                schedule.insert(i + 1, reheat_ts)
                n_reheats += 1
                if return_stats:
                    stats.reheat_positions.append(i)

        prev_x0 = x0.clone()

        # Step to next timestep in (possibly modified) schedule
        t_next = int(schedule[i + 1])
        a_n = ac[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

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
