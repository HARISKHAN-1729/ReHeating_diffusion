"""
Inception Score (IS) computation using Inception v3.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def compute_is(
    images: torch.Tensor,
    batch_size: int = 64,
    splits: int = 10,
    device: str = 'cuda',
) -> Tuple[float, float]:
    """
    Compute Inception Score.

    Args:
        images: (N, 3, H, W) in [-1, 1] or [0, 1] or [0, 255] uint8
        batch_size: batch size for Inception forward pass
        splits: number of splits for mean/std computation
        device: compute device

    Returns:
        (IS_mean, IS_std) tuple
    """
    from torchvision.models import inception_v3
    from torchvision import transforms

    # Normalize to [0, 1] if needed
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    elif images.min() < 0:
        images = (images + 1) / 2  # [-1,1] -> [0,1]

    # Load Inception v3
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Resize to 299x299 (Inception input)
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    # Normalize with ImageNet stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Get predictions
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        batch = up(batch)
        batch = torch.stack([normalize(img) for img in batch])

        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(batch)
        probs = F.softmax(logits, dim=1)
        preds.append(probs.cpu())

    preds = torch.cat(preds, dim=0).numpy()

    # Compute IS
    scores = []
    N = len(preds)
    split_size = N // splits

    for k in range(splits):
        part = preds[k * split_size:(k + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))
