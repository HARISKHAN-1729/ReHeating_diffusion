"""
Family 5: Adaptive reheat schedule.
Triggers reheating when the model's prediction changes sharply between steps.
The schedule itself is monotonic — the adaptive sampler inserts reheat steps on-the-fly.
"""

import numpy as np
from typing import Optional


def make(N: int = 100, num_train_timesteps: int = 1000, **kwargs) -> np.ndarray:
    """
    Returns a monotonic base schedule for adaptive sampling.
    The adaptive sampler (in samplers/ddim_reheating.py) modifies it on-the-fly.
    """
    from . import monotonic
    return monotonic.make(N=N, num_train_timesteps=num_train_timesteps)


def calibrate_threshold(
    model,
    alphas_cumprod,
    shape: tuple,
    N: int = 50,
    n_calibration: int = 100,
    num_train_timesteps: int = 1000,
    percentile: float = 80.0,
    device: str = 'cuda',
    **kwargs,
) -> float:
    """
    Calibrate the adaptive reheat threshold tau.

    Runs n_calibration monotonic trajectories and records how much the x0
    prediction changes between consecutive steps. The threshold is set at
    the given percentile of these changes — reheating triggers when the
    prediction changes more than most steps.

    Args:
        model: DDPMWrapper (callable: model(x_t, timestep) -> eps_pred)
        alphas_cumprod: model's own alphas_cumprod tensor (shape [T])
        shape: sample shape, e.g. (1, 3, 32, 32)
        N: number of steps for calibration trajectories
        n_calibration: number of trajectories to run
        num_train_timesteps: model's T
        percentile: which percentile to use as threshold
        device: compute device

    Returns:
        float: calibrated threshold tau
    """
    import torch
    from . import monotonic

    timesteps = monotonic.make(N=N, num_train_timesteps=num_train_timesteps)
    ts_tensor = torch.tensor(timesteps, dtype=torch.long, device=device)
    ac = alphas_cumprod.to(device)
    all_deltas = []

    with torch.no_grad():
        for _ in range(n_calibration):
            x = torch.randn((1, *shape[1:]), device=device)
            prev_x0 = None

            for i in range(len(ts_tensor) - 1):
                t_cur = ts_tensor[i]
                a_c = ac[t_cur]
                t_batch = t_cur.unsqueeze(0)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    eps = model(x, t_batch)
                eps = eps.float()

                x0 = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)

                if prev_x0 is not None:
                    delta = (x0 - prev_x0).pow(2).mean().sqrt().item()
                    all_deltas.append(delta)

                prev_x0 = x0.clone()

                t_next = ts_tensor[i + 1]
                a_n = ac[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
                eps_recon = (x - a_c.sqrt() * x0) / (1 - a_c).sqrt()
                x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps_recon

    tau = float(np.percentile(all_deltas, percentile))
    print(f"  Calibrated tau = {tau:.4f} (p{percentile}, {len(all_deltas)} measurements)")
    return tau
