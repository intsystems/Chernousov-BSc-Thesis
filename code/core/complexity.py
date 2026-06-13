"""Dataset complexity C(X) = log det(I + K).

Pipeline: embeddings (N, M) -> distance matrix -> kernel matrix -> C(X).
Each stage is a separate function so kernels and metrics are swappable.
The embedding itself (score- or trajectory-based) is produced upstream.

Bandwidth convention: lambda must NOT depend on the dataset being measured.
Calibrate it once with `calibrate_bandwidth` on an independent reference
sample from the model's training distribution, then reuse it for every
evaluation under that model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import pdist, squareform


def pairwise_distance(embeddings: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Pairwise distance matrix.

    Args:
        embeddings: (N, M) array
        metric: any metric accepted by scipy.spatial.distance.pdist

    Returns:
        D: (N, N) symmetric distance matrix
    """
    return squareform(pdist(embeddings, metric=metric))


def calibrate_bandwidth(reference_embeddings: np.ndarray,
                        metric: str = 'euclidean') -> float:
    """Bandwidth from an independent reference sample (median heuristic).

    Using a reference sample drawn from the model's training distribution
    makes lambda a property of the score model, not of the dataset being
    measured.

    Args:
        reference_embeddings: (N_ref, M) embeddings of the reference sample

    Returns:
        lambda_: scalar bandwidth, reused for all evaluations under this model
    """
    D = pairwise_distance(reference_embeddings, metric=metric)
    return median_heuristic(D)


def median_heuristic(D: np.ndarray) -> float:
    """lambda = 1 / median(D_ij^2) over the strict upper triangle.

    WARNING: applying this to the dataset being measured makes lambda depend
    on the input and breaks comparability across datasets.  Use
    `calibrate_bandwidth` on an independent reference sample instead.

    Args:
        D: (N, N) distance matrix

    Returns:
        lambda_: scalar bandwidth (1.0 for degenerate inputs)
    """
    upper = D[np.triu_indices_from(D, k=1)]
    if len(upper) == 0:
        return 1.0
    med_sq = np.median(upper ** 2)
    if med_sq == 0:
        return 1.0
    return 1.0 / med_sq


def gaussian_kernel(D: np.ndarray, lambda_: float) -> np.ndarray:
    """Gaussian (RBF) kernel K_ij = exp(-lambda * D_ij^2)."""
    return np.exp(-lambda_ * D ** 2)


def laplacian_kernel(D: np.ndarray, lambda_: float) -> np.ndarray:
    """Laplacian kernel K_ij = exp(-sqrt(lambda) * D_ij)."""
    return np.exp(-np.sqrt(lambda_) * D)


def complexity(K: np.ndarray) -> tuple[float, np.ndarray]:
    """C(X) = log det(I + K) via Cholesky factorization.

    Args:
        K: (N, N) PSD kernel matrix

    Returns:
        C: scalar complexity value
        eigenvalues: (N,) eigenvalues of K in decreasing order (clipped at 0),
            useful for spectrum analysis
    """
    N = K.shape[0]

    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(np.clip(eigenvalues, 0, None))[::-1]

    # I + K is positive definite, so Cholesky is safe and O(N^3 / 3)
    L = np.linalg.cholesky(np.eye(N) + K)
    C = 2.0 * np.sum(np.log(np.diag(L)))

    return C, eigenvalues


def compute_complexity(embeddings: np.ndarray, lambda_: float,
                       kernel_fn: Callable[[np.ndarray, float], np.ndarray] = gaussian_kernel,
                       ) -> dict:
    """Full pipeline: embeddings -> distances -> kernel -> C(X).

    Args:
        embeddings: (N, M) pre-computed embeddings (score, trajectory, ...)
        lambda_: kernel bandwidth; obtain via `calibrate_bandwidth` on an
            independent reference sample
        kernel_fn: callable(D, lambda_) -> K, default `gaussian_kernel`

    Returns:
        dict with keys 'C' (scalar), 'K' (N, N), 'D' (N, N),
        'eigenvalues' (N,), 'lambda' (bandwidth used)
    """
    D = pairwise_distance(embeddings)
    K = kernel_fn(D, lambda_)
    C, eigenvalues = complexity(K)

    return {
        'C': C,
        'K': K,
        'D': D,
        'eigenvalues': eigenvalues,
        'lambda': lambda_,
    }


def score_embedding(score_vectors: np.ndarray) -> np.ndarray:
    """Flatten per-level score vectors (N, L, D) into embeddings (N, L*D)."""
    N = score_vectors.shape[0]
    return score_vectors.reshape(N, -1)
