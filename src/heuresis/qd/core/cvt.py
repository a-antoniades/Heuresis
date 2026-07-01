"""Centroidal Voronoi Tessellation utilities for MAP-Elites.

Ported from science-codeevolve/src/codeevolve/utils/cvt_utils.py (Apache 2.0).
"""

from __future__ import annotations

import numpy as np


def closest_centroid_idx(point: np.ndarray, centroids: np.ndarray) -> int:
    """Finds the index of the closest centroid to a given point.

    Args:
        point: A 1D array representing the coordinates of the point.
        centroids: A 2D array where each row is a centroid.

    Returns:
        The integer index of the closest centroid.
    """
    dist_to_centroids: np.ndarray = np.sum((centroids - point) ** 2, axis=1)
    return np.argmin(dist_to_centroids).item()


def cvt(
    num_centroids: int,
    num_samples: int,
    feature_bounds: list[tuple[float, float]],
    max_iter: int = 300,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Generates centroids using Centroidal Voronoi Tessellation (Lloyd's algorithm).

    Args:
        num_centroids: The number of centroids (k) to generate.
        num_samples: The number of random points for partitioning the space.
        feature_bounds: List of (min_val, max_val) per feature dimension.
        max_iter: Maximum iterations.
        tolerance: Convergence threshold — stops if max centroid shift is below this.

    Returns:
        Array of shape (num_centroids, num_features) with final centroid positions.
    """
    num_features = len(feature_bounds)
    samples: np.ndarray = np.array(
        [
            [
                np.random.uniform(feature_bounds[i][0], feature_bounds[i][1])
                for i in range(num_features)
            ]
            for j in range(num_centroids + num_samples)
        ],
        dtype=np.float64,
    )

    centroids: np.ndarray = samples[:num_centroids, :].copy()
    points: np.ndarray = samples[num_centroids : num_centroids + num_samples, :]

    for iteration in range(max_iter):
        prev_centroids: np.ndarray = centroids.copy()

        centroid2points: list[list[int]] = [[] for _ in range(num_centroids)]
        for i in range(num_samples):
            centroid_idx: int = closest_centroid_idx(points[i, :], centroids)
            centroid2points[centroid_idx].append(i)

        for j in range(num_centroids):
            if centroid2points[j]:
                centroids[j] = np.mean(points[centroid2points[j], :], axis=0)

        centroid_shift: float = np.max(np.linalg.norm(centroids - prev_centroids, axis=1))
        if centroid_shift < tolerance:
            return centroids

    return centroids
