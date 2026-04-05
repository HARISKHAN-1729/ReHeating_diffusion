"""
HuggingFace DDPM wrapper for CIFAR-10.
Loads google/ddpm-cifar10-32 and exposes a clean interface:
  model(x_t, timestep_int) -> eps_pred
"""

import os
import torch
import torch.nn as nn
from typing import Optional


_CACHED_MODEL = None


class DDPMWrapper(nn.Module):
    """
    Wrapper around HuggingFace DDPM UNet.
    Provides: model(x_t, timestep_int) -> eps_pred
    """

    def __init__(self, model_id: str = "google/ddpm-cifar10-32", cache_dir: str = "checkpoints"):
        super().__init__()
        from diffusers import DDPMPipeline
        os.makedirs(cache_dir, exist_ok=True)
        pipe = DDPMPipeline.from_pretrained(model_id, cache_dir=cache_dir)
        self.unet = pipe.unet
        self.unet.eval()
        # Store scheduler info
        self.num_train_timesteps = pipe.scheduler.config.num_train_timesteps

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (batch, 3, 32, 32) noisy image in [-1, 1]
            timestep: (batch,) integer timestep [0, 999]

        Returns:
            eps_pred: (batch, 3, 32, 32) predicted noise
        """
        # UNet expects timestep as long tensor
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(x_t.shape[0])
        out = self.unet(x_t, timestep)
        # HuggingFace returns UNet2DOutput with .sample attribute
        if hasattr(out, 'sample'):
            return out.sample
        return out


def load_ddpm(
    model_id: str = "google/ddpm-cifar10-32",
    device: str = "cuda",
    cache_dir: str = "checkpoints",
) -> DDPMWrapper:
    """
    Load DDPM model with caching.

    Returns:
        DDPMWrapper on the specified device.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL.to(device)

    print(f"Loading DDPM model: {model_id}")
    model = DDPMWrapper(model_id=model_id, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()
    _CACHED_MODEL = model
    print(f"DDPM loaded. UNet params: {sum(p.numel() for p in model.unet.parameters()):,}")
    return model
