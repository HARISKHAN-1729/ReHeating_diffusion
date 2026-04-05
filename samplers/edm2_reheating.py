"""
EDM2-style sampler with non-monotonic sigma schedule.
Supports both Euler and Heun (2nd-order) stepping.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple


@torch.no_grad()
def edm2_sample(
    denoiser,
    sigmas: np.ndarray,
    shape: tuple,
    device: str = 'cuda',
    S_churn: float = 0.0,
    S_noise: float = 1.0,
    order: int = 1,
    return_trajectory: bool = False,
    use_fp16: bool = True,
) -> Union[Tensor, Tuple[Tensor, list]]:
    """
    EDM2-style sampler with non-monotonic sigma schedule.

    The key insight: d = (x - D(x,sigma)) / sigma is the update direction.
    x_next = x + (sigma_next - sigma_cur) * d
    When sigma_next > sigma_cur, this adds noise (reheat step).

    Args:
        denoiser: D(x, sigma) -> denoised x
        sigmas: from make_edm2_schedule(), includes terminal 0
        shape: (batch_size, channels, height, width)
        device: compute device
        S_churn: Karras stochastic sampling (0 = deterministic)
        S_noise: noise multiplier for stochastic sampling
        order: 1=Euler, 2=Heun
        return_trajectory: whether to return intermediate states
        use_fp16: use float16 for model forward passes

    Returns:
        Generated samples.
    """
    sig = torch.tensor(sigmas, dtype=torch.float32, device=device)
    x = torch.randn(shape, device=device) * sig[0]
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(sig) - 1):
        s_cur = sig[i]
        s_next = sig[i + 1]

        if s_cur == 0:
            break

        # Optional stochastic injection (Karras EDM2)
        if S_churn > 0 and s_next > 0:
            gamma = min(S_churn / (len(sig) - 1), 2**0.5 - 1)
            s_hat = s_cur * (1 + gamma)
            x = x + (s_hat**2 - s_cur**2).sqrt() * S_noise * torch.randn_like(x)
            s_cur = s_hat

        # Evaluate denoiser
        if use_fp16 and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                d_cur = denoiser(x, s_cur.expand(shape[0]))
            d_cur = d_cur.float()
        else:
            d_cur = denoiser(x, s_cur.expand(shape[0]))

        # EDM2 direction: (x - D(x,sigma)) / sigma
        d = (x - d_cur) / s_cur

        if order == 2 and s_next > 0:
            # Heun's method (2nd order)
            x_next = x + (s_next - s_cur) * d

            if use_fp16 and device != 'cpu':
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    d_next_denoised = denoiser(x_next, s_next.expand(shape[0]))
                d_next_denoised = d_next_denoised.float()
            else:
                d_next_denoised = denoiser(x_next, s_next.expand(shape[0]))

            d_next = (x_next - d_next_denoised) / s_next
            d_avg = (d + d_next) / 2
            x = x + (s_next - s_cur) * d_avg
        else:
            # Euler step (works for both directions)
            x = x + (s_next - s_cur) * d

        if return_trajectory:
            traj.append(x.cpu().clone())

    # Final denoised output
    if use_fp16 and device != 'cpu':
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x_final = denoiser(x, torch.zeros(shape[0], device=device) + sig[-2])
        x_final = x_final.float()
    else:
        x_final = d_cur  # Last denoiser output

    if return_trajectory:
        return x_final, traj
    return x_final
