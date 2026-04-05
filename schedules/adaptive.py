"""
Family 5: Adaptive reheat schedule.
Triggers reheating when the model signals uncertainty (prediction instability).
The most principled variant — reheats only when needed.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class AdaptiveStats:
    """Diagnostics from adaptive sampling."""
    reheat_positions: List[int]
    prediction_deltas: List[float]
    total_nfe: int
    total_reheats: int


def make(N: int = 100, **kwargs) -> np.ndarray:
    """
    Returns a monotonic base schedule for adaptive sampling.
    The actual adaptive behavior happens in the sampler, not the schedule.
    This provides the base schedule that the adaptive sampler modifies on-the-fly.
    """
    from .monotonic import make as make_monotonic
    return make_monotonic(N=N)


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
    Calibrate the adaptive reheat threshold tau.

    Runs n_calibration monotonic samples and computes the percentile
    of prediction-change magnitudes. Large changes indicate the model
    is uncertain / near a mode boundary.

    Args:
        model: UNet that takes (x_t, timestep_int) -> eps_pred
        shape: Sample shape (batch, channels, height, width)
        alpha_bars: Base monotonic schedule. If None, uses cosine with N=50.
        n_calibration: Number of calibration trajectories.
        percentile: Which percentile to use as threshold.
        device: Device for computation.
        num_train_timesteps: Model's training timestep count.

    Returns:
        Calibrated threshold tau.
    """
    if alpha_bars is None:
        alpha_bars = make(N=50)

    ab = torch.tensor(alpha_bars, dtype=torch.float32, device=device)
    all_deltas = []

    for _ in range(n_calibration):
        x = torch.randn((1, *shape[1:]), device=device)
        prev_x0 = None

        for i in range(len(ab) - 1):
            a_c = ab[i]
            ts = ((1 - a_c) * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)
            ts = ts.unsqueeze(0)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                eps = model(x, ts)
            eps = eps.float()

            x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)

            if prev_x0 is not None:
                delta = (x0 - prev_x0).pow(2).mean().sqrt().item()
                all_deltas.append(delta)

            prev_x0 = x0.clone()

            a_n = ab[i + 1]
            eps_recon = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()
            x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps_recon

    tau = float(np.percentile(all_deltas, percentile))
    print(f"Calibrated tau = {tau:.4f} (percentile {percentile}, "
          f"from {len(all_deltas)} measurements)")
    return tau
