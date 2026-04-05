"""
EDM2 wrapper for CIFAR-10 and ImageNet-64.

Attempts to load pretrained EDM2 checkpoint from Karras et al.
If unavailable, provides a stub that mimics the interface.

Interface: denoiser(x, sigma) -> denoised_x
Handles EDM2 preconditioning internally.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class EDM2Stub(nn.Module):
    """
    Stub EDM2 denoiser for when checkpoints are unavailable.
    Implements the correct interface but outputs meaningless results.
    Use for pipeline verification only.
    """

    def __init__(self, img_channels: int = 3, img_resolution: int = 32):
        super().__init__()
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        # Minimal conv to have some parameters
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, img_channels, 3, padding=1),
        )
        self._is_stub = True

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        EDM2 denoiser interface: D(x, sigma) -> denoised x.

        Uses EDM2 preconditioning:
          D(x;sigma) = c_skip(sigma)*x + c_out(sigma)*F(c_in(sigma)*x; c_noise(sigma))
        """
        # EDM2 preconditioning coefficients
        sigma_data = 0.5
        c_skip = sigma_data ** 2 / (sigma.view(-1, 1, 1, 1) ** 2 + sigma_data ** 2)
        c_out = sigma.view(-1, 1, 1, 1) * sigma_data / (sigma.view(-1, 1, 1, 1) ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma.view(-1, 1, 1, 1) ** 2 + sigma_data ** 2).sqrt()

        # Network forward
        f_out = self.net(c_in * x)

        return c_skip * x + c_out * f_out


def load_edm2(
    dataset: str = "cifar10",
    device: str = "cuda",
    cache_dir: str = "checkpoints",
) -> nn.Module:
    """
    Load EDM2 model. Tries pretrained checkpoint first, falls back to stub.

    Args:
        dataset: "cifar10" or "imagenet64"
        device: compute device
        cache_dir: directory for checkpoints

    Returns:
        EDM2 denoiser module.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if dataset == "cifar10":
        img_channels, img_resolution = 3, 32
    elif dataset == "imagenet64":
        img_channels, img_resolution = 3, 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Try to load pretrained checkpoint
    checkpoint_paths = [
        os.path.join(cache_dir, f"edm2-{dataset}.pkl"),
        os.path.join(cache_dir, f"edm2-{dataset}.pt"),
    ]

    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            try:
                print(f"Loading EDM2 checkpoint from {ckpt_path}")
                if ckpt_path.endswith('.pkl'):
                    import pickle
                    with open(ckpt_path, 'rb') as f:
                        model = pickle.load(f)['ema']
                else:
                    model = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = model.to(device).eval()
                print(f"EDM2 loaded from {ckpt_path}")
                return model
            except Exception as e:
                print(f"Failed to load {ckpt_path}: {e}")

    # Fall back to stub
    print(f"EDM2 checkpoint not available for {dataset}. Using stub for pipeline verification.")
    print("  To use real EDM2, download checkpoint to checkpoints/edm2-cifar10.pkl")
    model = EDM2Stub(img_channels=img_channels, img_resolution=img_resolution)
    model = model.to(device).eval()
    return model
