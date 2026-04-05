"""
Mode coverage metrics for 2D toy experiments.
- Count modes covered: mode is "covered" if >= 1% of samples within 2*sigma
- Wasserstein-2 distance to ground truth (using POT library)
"""

import numpy as np
from typing import Tuple


def count_modes(
    samples: np.ndarray,
    centers: np.ndarray,
    stds: np.ndarray,
    threshold: float = 0.01,
    radius_factor: float = 2.0,
) -> int:
    """
    Count how many modes are covered by generated samples.

    A mode is "covered" if >= threshold fraction of samples fall within
    radius_factor * std of its center.

    Args:
        samples: (N, 2) generated samples
        centers: (K, 2) mode centers
        stds: (K,) standard deviations per mode
        threshold: minimum fraction of samples for coverage
        radius_factor: multiplier for the coverage radius

    Returns:
        Number of modes covered (int).
    """
    n_samples = len(samples)
    n_modes = len(centers)
    covered = 0

    for i in range(n_modes):
        distances = np.sqrt(np.sum((samples - centers[i]) ** 2, axis=1))
        radius = radius_factor * stds[i]
        n_within = np.sum(distances <= radius)
        fraction = n_within / n_samples
        if fraction >= threshold:
            covered += 1

    return covered


def wasserstein2(
    samples: np.ndarray,
    reference: np.ndarray,
    n_max: int = 5000,
) -> float:
    """
    Compute Wasserstein-2 distance between two 2D distributions.

    Args:
        samples: (N, 2) generated samples
        reference: (M, 2) reference samples
        n_max: max samples to use (for computational tractability)

    Returns:
        W2 distance (float).
    """
    try:
        import ot
    except ImportError:
        print("POT library not installed. Computing L2-based approximation.")
        return _approx_w2(samples, reference)

    # Subsample if too many
    if len(samples) > n_max:
        idx = np.random.choice(len(samples), n_max, replace=False)
        samples = samples[idx]
    if len(reference) > n_max:
        idx = np.random.choice(len(reference), n_max, replace=False)
        reference = reference[idx]

    n, m = len(samples), len(reference)
    a = np.ones(n) / n
    b = np.ones(m) / m

    M = ot.dist(samples, reference, metric='sqeuclidean')
    w2_sq = ot.emd2(a, b, M)

    return float(np.sqrt(max(w2_sq, 0)))


def _approx_w2(samples, reference):
    """Approximate W2 using mean and covariance (Gaussian assumption)."""
    mu1, mu2 = samples.mean(axis=0), reference.mean(axis=0)
    cov1 = np.cov(samples.T)
    cov2 = np.cov(reference.T)

    # Bures metric (exact for Gaussians)
    diff_mu = np.sum((mu1 - mu2) ** 2)

    from scipy.linalg import sqrtm
    sqrt_cov1 = sqrtm(cov1)
    M = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    if np.iscomplexobj(M):
        M = M.real
    bures = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(M)

    return float(np.sqrt(max(diff_mu + bures, 0)))


def mode_assignment(
    samples: np.ndarray,
    centers: np.ndarray,
    stds: np.ndarray,
    radius_factor: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each sample to its nearest mode (or -1 if not within radius).

    Returns:
        assignments: (N,) mode index or -1
        counts: (K,) number of samples per mode
    """
    n_samples = len(samples)
    n_modes = len(centers)

    # Compute distances to all centers
    dists = np.sqrt(np.sum(
        (samples[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
        axis=2
    ))  # (N, K)

    nearest = np.argmin(dists, axis=1)  # (N,)
    nearest_dist = dists[np.arange(n_samples), nearest]

    # Check if within radius
    radii = stds[nearest] * radius_factor
    assignments = np.where(nearest_dist <= radii, nearest, -1)

    counts = np.zeros(n_modes, dtype=int)
    for i in range(n_modes):
        counts[i] = np.sum(assignments == i)

    return assignments, counts
