"""
Flow matching Euler sampler with non-monotonic time schedule.
Backward steps (t decreasing) use negative dt — valid Euler step.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple


@torch.no_grad()
def flow_sample(
    velocity_model,
    times: np.ndarray,
    shape: tuple,
    device: str = 'cuda',
    return_trajectory: bool = False,
    use_fp16: bool = True,
) -> Union[Tensor, Tuple[Tensor, list]]:
    """
    Flow matching Euler sampler with non-monotonic time schedule.

    Euler step: x = x + dt * v_theta(x, t)
    Backward step = negative dt (reheat).

    Args:
        velocity_model: v(x, t) -> velocity field
        times: from make_flow_schedule(), ordered noise->data (low t -> high t)
        shape: (batch_size, channels, height, width)
        device: compute device
        return_trajectory: whether to return intermediate states
        use_fp16: use float16 for model forward passes

    Returns:
        Generated samples.
    """
    t_arr = torch.tensor(times, dtype=torch.float32, device=device)
    x = torch.randn(shape, device=device)
    traj = [x.cpu().clone()] if return_trajectory else None

    for i in range(len(t_arr) - 1):
        t_cur = t_arr[i]
        t_next = t_arr[i + 1]
        dt = t_next - t_cur  # Can be negative (reheat)

        if use_fp16 and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                v = velocity_model(x, t_cur.expand(shape[0]))
            v = v.float()
        else:
            v = velocity_model(x, t_cur.expand(shape[0]))

        x = x + dt * v

        if return_trajectory:
            traj.append(x.cpu().clone())

    if return_trajectory:
        return x, traj
    return x
