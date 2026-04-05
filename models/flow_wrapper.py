"""
OT-CFM (Optimal Transport Conditional Flow Matching) wrapper.

Provides:
- Small UNet for training from scratch on CIFAR-10
- Training loop with OT-CFM objective
- Interface: velocity_model(x, t) -> v

Training: OT-CFM loss with linear interpolation x_t = (1-t)*eps + t*x_0
Velocity target: v = x_0 - eps
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class SmallUNet(nn.Module):
    """
    Small UNet (~5M params) for flow matching on CIFAR-10.
    Predicts velocity field v(x, t).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = nn.Sequential(
            ResBlock(base_channels, time_dim),
            ResBlock(base_channels, time_dim),
        )
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.down2 = nn.Sequential(
            ResBlock(base_channels * 2, time_dim),
            ResBlock(base_channels * 2, time_dim),
        )
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid = nn.Sequential(
            ResBlock(base_channels * 4, time_dim),
            ResBlock(base_channels * 4, time_dim),
        )

        # Decoder
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.Sequential(
            ResBlock(base_channels * 4, time_dim),  # skip connection doubles channels
            ResBlock(base_channels * 4, time_dim),
        )
        self.up2_proj = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = nn.Sequential(
            ResBlock(base_channels * 2, time_dim),  # skip connection
            ResBlock(base_channels * 2, time_dim),
        )
        self.up1_proj = nn.Conv2d(base_channels * 2, base_channels, 1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 32, 32) image
            t: (batch,) time values in [0, 1]

        Returns:
            v: (batch, 3, 32, 32) predicted velocity
        """
        t_emb = self.time_embed(t)

        # Encoder
        h = self.conv_in(x)
        h1 = h
        for block in self.down1:
            h1 = block(h1, t_emb)
        skip1 = h1

        h2 = self.downsample1(h1)
        for block in self.down2:
            h2 = block(h2, t_emb)
        skip2 = h2

        # Bottleneck
        h = self.downsample2(h2)
        for block in self.mid:
            h = block(h, t_emb)

        # Decoder
        h = self.upsample2(h)
        h = torch.cat([h, skip2], dim=1)
        for block in self.up2:
            h = block(h, t_emb)
        h = self.up2_proj(h)

        h = self.upsample1(h)
        h = torch.cat([h, skip1], dim=1)
        for block in self.up1:
            h = block(h, t_emb)
        h = self.up1_proj(h)

        return self.conv_out(h)


class FlowWrapper(nn.Module):
    """Wrapper to normalize the velocity model interface."""

    def __init__(self, unet: SmallUNet):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """velocity_model(x, t) -> v"""
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        return self.unet(x, t)


def train_flow_model(
    n_steps: int = 50000,
    batch_size: int = 128,
    lr: float = 2e-4,
    device: str = 'cuda',
    save_path: str = 'checkpoints/flow_cifar10.pt',
) -> FlowWrapper:
    """
    Train flow matching model on CIFAR-10.

    Uses OT-CFM objective:
      x_t = (1-t)*noise + t*x_0
      v_target = x_0 - noise
      loss = ||v_theta(x_t, t) - v_target||^2
    """
    import torchvision
    import torchvision.transforms as T

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load CIFAR-10
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='checkpoints/data', train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    model = SmallUNet(in_channels=3, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training flow model: {n_params:,} parameters, {n_steps} steps")

    model.train()
    data_iter = iter(dataloader)
    losses = []
    pbar = tqdm(range(n_steps), desc="Training flow model")

    for step in pbar:
        try:
            x_0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_0, _ = next(data_iter)

        x_0 = x_0.to(device)
        noise = torch.randn_like(x_0)
        t = torch.rand(batch_size, device=device)

        # OT-CFM interpolation
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1 - t_view) * noise + t_view * x_0

        # Velocity target
        v_target = x_0 - noise

        # Forward
        v_pred = model(x_t, t)
        loss = (v_pred - v_target).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 500 == 0:
            avg = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses)
            pbar.set_postfix(loss=f"{avg:.4f}")

    wrapper = FlowWrapper(model)
    torch.save(model.state_dict(), save_path)
    print(f"Saved flow model to {save_path}")
    return wrapper


def load_flow_model(
    path: str = 'checkpoints/flow_cifar10.pt',
    device: str = 'cuda',
) -> FlowWrapper:
    """Load a trained flow matching model."""
    model = SmallUNet(in_channels=3, base_channels=64)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    wrapper = FlowWrapper(model).to(device)
    wrapper.eval()
    return wrapper
