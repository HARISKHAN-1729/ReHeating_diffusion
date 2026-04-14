"""
HuggingFace DDPM wrapper for CIFAR-10.
Loads google/ddpm-cifar10-32 and exposes a clean interface:
  model(x_t, timestep_int) -> eps_pred
  model.alphas_cumprod       -> the model's own noise schedule (for DDIM formula)
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
    Also stores: model.alphas_cumprod for use in the DDIM sampling formula.
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
        # CRITICAL FIX: store the model's own alpha schedule.
        # The DDIM formula MUST use these values — not independently constructed ones.
        # alphas_cumprod[t] = product of (1 - beta_s) for s=0..t
        # alphas_cumprod[0]   ≈ 0.9999  (almost clean)
        # alphas_cumprod[999] ≈ 0.0047  (almost pure noise)
        self.alphas_cumprod = pipe.scheduler.alphas_cumprod.clone()

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (batch, 3, 32, 32) noisy image in [-1, 1]
            timestep: (batch,) integer timestep [0, 999]

        Returns:
            eps_pred: (batch, 3, 32, 32) predicted noise
        """
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(x_t.shape[0])
        out = self.unet(x_t, timestep)
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
        Access model.alphas_cumprod for the noise schedule.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL.to(device)

    print(f"Loading DDPM model: {model_id}")
    model = DDPMWrapper(model_id=model_id, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()
    # Keep alphas_cumprod as float32
    model.alphas_cumprod = model.alphas_cumprod.float()
    _CACHED_MODEL = model
    print(f"DDPM loaded. UNet params: {sum(p.numel() for p in model.unet.parameters()):,}")
    print(f"  alphas_cumprod[0]   = {model.alphas_cumprod[0]:.6f}  (should be ~0.9999)")
    print(f"  alphas_cumprod[499] = {model.alphas_cumprod[499]:.6f}  (should be ~0.08)")
    print(f"  alphas_cumprod[999] = {model.alphas_cumprod[999]:.6f}  (should be ~0.005)")
    return model
