"""
FID computation using clean-fid or pytorch-fid.
Supports FID-50K (full) and FID-10K (fast iteration).
"""

import os
import numpy as np
import torch
from pathlib import Path


def save_images_for_fid(images: torch.Tensor, output_dir: str, start_idx: int = 0):
    """
    Save a batch of images to disk for FID computation.

    Args:
        images: (batch, 3, H, W) in [-1, 1] float or [0, 255] uint8
        output_dir: directory to save PNG images
        start_idx: starting index for filenames
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    if images.dtype == torch.float32 or images.dtype == torch.float16:
        # Convert from [-1, 1] to [0, 255] uint8
        images = ((images.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)

    images = images.cpu().numpy()

    for i, img in enumerate(images):
        # (C, H, W) -> (H, W, C)
        img = img.transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img.squeeze(-1)
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(output_dir, f"{start_idx + i:06d}.png"))


def compute_fid(
    generated_dir: str,
    dataset_name: str = "cifar10",
    batch_size: int = 64,
    device: str = "cuda",
    mode: str = "clean",
) -> float:
    """
    Compute FID between generated images and reference dataset.

    Args:
        generated_dir: path to directory of generated PNG images
        dataset_name: "cifar10" or "imagenet64"
        batch_size: batch size for feature extraction
        device: device for computation
        mode: "clean" (clean-fid) or "pytorch" (pytorch-fid)

    Returns:
        FID score (float).
    """
    if mode == "clean":
        return _compute_fid_clean(generated_dir, dataset_name, batch_size, device)
    else:
        return _compute_fid_pytorch(generated_dir, dataset_name, batch_size)


def _compute_fid_clean(generated_dir, dataset_name, batch_size, device):
    """Compute FID using clean-fid library."""
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("clean-fid not installed. Trying pytorch-fid...")
        return _compute_fid_pytorch(generated_dir, dataset_name, batch_size)

    if dataset_name == "cifar10":
        score = cleanfid.compute_fid(
            generated_dir,
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="train",
            batch_size=batch_size,
            device=torch.device(device),
        )
    else:
        # For other datasets, need custom reference stats
        ref_dir = f"checkpoints/fid_stats/{dataset_name}"
        if os.path.exists(ref_dir):
            score = cleanfid.compute_fid(generated_dir, ref_dir, batch_size=batch_size,
                                         device=torch.device(device))
        else:
            print(f"No reference stats for {dataset_name}. Computing FID against generated only.")
            score = float('nan')

    return score


def _compute_fid_pytorch(generated_dir, dataset_name, batch_size):
    """Compute FID using pytorch-fid."""
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("Neither clean-fid nor pytorch-fid available.")
        return float('nan')

    # pytorch-fid needs two directories
    if dataset_name == "cifar10":
        ref_dir = "checkpoints/fid_stats/cifar10_train"
        if not os.path.exists(ref_dir):
            _prepare_cifar10_reference(ref_dir)
        score = fid_score.calculate_fid_given_paths(
            [generated_dir, ref_dir],
            batch_size, 'cuda', 2048
        )
    else:
        print(f"No reference for {dataset_name}")
        score = float('nan')

    return score


def _prepare_cifar10_reference(output_dir: str):
    """Save CIFAR-10 training set as PNGs for FID reference."""
    import torchvision
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(
        root='checkpoints/data', train=True, download=True
    )

    print(f"Saving CIFAR-10 training set to {output_dir}...")
    for i, (img, _) in enumerate(dataset):
        img.save(os.path.join(output_dir, f"{i:06d}.png"))
    print(f"Saved {len(dataset)} images")


def compute_fid_from_tensors(
    generated: torch.Tensor,
    dataset_name: str = "cifar10",
    batch_size: int = 64,
    device: str = "cuda",
    tmp_dir: str = "results/tmp_fid",
) -> float:
    """
    Convenience: compute FID directly from a tensor of generated images.

    Args:
        generated: (N, 3, H, W) in [-1, 1]
        dataset_name: reference dataset
        batch_size: for feature extraction
        device: compute device
        tmp_dir: temporary directory for saving images

    Returns:
        FID score
    """
    import shutil

    os.makedirs(tmp_dir, exist_ok=True)
    save_images_for_fid(generated, tmp_dir)

    score = compute_fid(tmp_dir, dataset_name, batch_size, device)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return score
