"""
4-layer MLP for 2D toy experiments.
Input: (x in R^2, t in R) -> Output: eps in R^2 (noise prediction)
Architecture: [x; sinusoidal_embed(t)] -> 256 -> 256 -> 256 -> 256 -> 2
~50K parameters. Training on 8-Gaussian ring mixture.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep t."""

    def __init__(self, dim: int = 64, max_freq: float = 10000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), half))
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) -> (batch, dim)"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch, 1)
        args = t * self.freqs  # (batch, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ToyMLP(nn.Module):
    """4-layer MLP noise predictor for 2D data."""

    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, time_emb_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(dim=time_emb_dim)
        input_dim = data_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) noisy data
            t: (batch,) or scalar — integer timestep or alpha_bar value

        Returns:
            eps_pred: (batch, 2) predicted noise
        """
        t_emb = self.time_embed(t.float())
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


def make_8_gaussians(n_samples: int = 100000, std: float = 0.05) -> np.ndarray:
    """Generate samples from 8-Gaussian ring mixture."""
    centers = []
    for i in range(8):
        angle = 2 * math.pi * i / 8
        centers.append([math.cos(angle), math.sin(angle)])
    centers = np.array(centers)

    # Assign samples to modes uniformly
    assignments = np.random.randint(0, 8, n_samples)
    samples = centers[assignments] + np.random.randn(n_samples, 2) * std
    return samples.astype(np.float32)


def get_8_gaussian_centers(std: float = 0.05):
    """Return the 8 mode centers and standard deviations."""
    centers = []
    for i in range(8):
        angle = 2 * math.pi * i / 8
        centers.append([math.cos(angle), math.sin(angle)])
    return np.array(centers, dtype=np.float32), np.full(8, std, dtype=np.float32)


def _cosine_alpha_bar(t, s=0.008):
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return f / f[0]


def train_toy_mlp(
    n_steps: int = 50000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = 'cuda',
    save_path: str = 'checkpoints/toy_mlp.pt',
    num_train_timesteps: int = 1000,
) -> ToyMLP:
    """
    Train the toy MLP with standard DDPM loss on 8-Gaussian ring mixture.

    Args:
        n_steps: Number of training steps.
        batch_size: Training batch size.
        lr: Learning rate.
        device: Device.
        save_path: Where to save the trained model.
        num_train_timesteps: Number of discrete timesteps for training.

    Returns:
        Trained ToyMLP model.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Pre-compute alpha_bar for all timesteps
    t_vals = np.linspace(0, 1, num_train_timesteps)
    alpha_bars = _cosine_alpha_bar(t_vals)
    alpha_bars = torch.tensor(alpha_bars, dtype=torch.float32, device=device)

    # Generate training data
    data = make_8_gaussians(n_samples=200000)
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    model = ToyMLP(data_dim=2, hidden_dim=256, time_emb_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training toy MLP: {sum(p.numel() for p in model.parameters())} parameters")

    model.train()
    losses = []
    pbar = tqdm(range(n_steps), desc="Training toy MLP")
    for step in pbar:
        # Sample batch
        idx = torch.randint(0, len(data_tensor), (batch_size,))
        x0 = data_tensor[idx]

        # Sample random timesteps
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
        ab = alpha_bars[t]  # (batch,)

        # Forward diffusion
        noise = torch.randn_like(x0)
        x_t = ab.unsqueeze(-1).sqrt() * x0 + (1 - ab.unsqueeze(-1)).sqrt() * noise

        # Predict noise
        eps_pred = model(x_t, t)

        # MSE loss
        loss = (eps_pred - noise).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 1000 == 0:
            avg_loss = np.mean(losses[-1000:]) if len(losses) >= 1000 else np.mean(losses)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved toy MLP to {save_path} (final loss: {np.mean(losses[-1000:]):.4f})")
    return model


def load_toy_mlp(path: str = 'checkpoints/toy_mlp.pt', device: str = 'cuda') -> ToyMLP:
    """Load a trained toy MLP."""
    model = ToyMLP(data_dim=2, hidden_dim=256, time_emb_dim=64)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model
