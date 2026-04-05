"""
Improved Precision and Recall (Kynkaanniemi et al., 2019).
Uses k-nearest neighbors in Inception feature space.
"""

import numpy as np
import torch
from typing import Tuple


def compute_features(
    images: torch.Tensor,
    batch_size: int = 64,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Extract Inception v3 features (pool3 layer, 2048-dim).

    Args:
        images: (N, 3, H, W) in [-1, 1]
        batch_size: batch size for forward pass
        device: compute device

    Returns:
        features: (N, 2048) numpy array
    """
    from torchvision.models import inception_v3
    from torchvision import transforms

    # Normalize
    if images.min() < 0:
        images = (images + 1) / 2

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Hook to get pool3 features
    features_list = []

    def hook_fn(module, input, output):
        features_list.append(output.squeeze(-1).squeeze(-1).cpu())

    # Register hook on avgpool layer
    handle = model.avgpool.register_forward_hook(hook_fn)

    up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            batch = up(batch)
            batch = torch.stack([normalize(img) for img in batch])
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(batch)

    handle.remove()
    features = torch.cat(features_list, dim=0).numpy()
    return features


def compute_precision_recall(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    k: int = 3,
) -> Tuple[float, float]:
    """
    Compute improved precision and recall.

    Args:
        real_features: (N_real, D) real image features
        gen_features: (N_gen, D) generated image features
        k: number of nearest neighbors

    Returns:
        (precision, recall) tuple
    """
    from sklearn.neighbors import NearestNeighbors

    # Fit KNN on real features
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_features)
    real_distances, _ = nn_real.kneighbors(real_features)
    real_radii = real_distances[:, -1]  # k-th nearest neighbor distance

    # Fit KNN on generated features
    nn_gen = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(gen_features)
    gen_distances, _ = nn_gen.kneighbors(gen_features)
    gen_radii = gen_distances[:, -1]

    # Precision: fraction of generated samples within real manifold
    distances_gen_to_real, _ = nn_real.kneighbors(gen_features)
    precision = float(np.mean(distances_gen_to_real[:, 0] <= real_radii[np.argmin(distances_gen_to_real, axis=1)]))

    # Simplified: check if each gen sample is within any real sample's radius
    dist_gen_to_real_all, idx_gen_to_real = nn_real.kneighbors(gen_features)
    in_manifold = np.zeros(len(gen_features), dtype=bool)
    for i in range(len(gen_features)):
        for j in range(k):
            if dist_gen_to_real_all[i, j] <= real_radii[idx_gen_to_real[i, j]]:
                in_manifold[i] = True
                break
    precision = float(np.mean(in_manifold))

    # Recall: fraction of real samples within generated manifold
    dist_real_to_gen_all, idx_real_to_gen = nn_gen.kneighbors(real_features)
    covered = np.zeros(len(real_features), dtype=bool)
    for i in range(len(real_features)):
        for j in range(k):
            if dist_real_to_gen_all[i, j] <= gen_radii[idx_real_to_gen[i, j]]:
                covered[i] = True
                break
    recall = float(np.mean(covered))

    return precision, recall
