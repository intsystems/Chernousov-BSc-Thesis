"""Gaussian kernel vs linear kernel for C(X) = log det(I + K).

Gaussian: K_ij = exp(-lambda * D_ij^2)   (distance-based, needs CND)
Linear:   K   = Phi Phi^T / c            (Gram-based, always PSD)

Both kernels should agree qualitatively on low-dimensional synthetic data;
the linear kernel is also usable in high dimension where the Gaussian
kernel saturates.

Sub-experiments:
  1. growth rate (N-cluster vs single-cluster), D=2
  2. growth rate, D=100
  3. varying separation, D=2

Outputs: results/linear_vs_gaussian_{D2,D100,separation}.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model import AnalyticScoreModel, VPNoiseSchedule
from core.trajectory import ode_trajectories, trajectory_embedding
from core.complexity import calibrate_bandwidth, compute_complexity
from common import gaussian_reference_embeddings

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SCHEDULE = VPNoiseSchedule()
T_EVAL = np.linspace(0.01, 0.99, 20)
CLUSTER_SPACING = 20.0


def C_linear(emb: np.ndarray, c: float = 1.0) -> float:
    """C(X) = log det(I + Phi Phi^T / c) via eigenvalues of the Gram matrix."""
    K = emb @ emb.T / c
    eigvals = np.clip(np.linalg.eigvalsh(K), 0, None)
    return np.sum(np.log(1 + eigvals))


def C_gaussian(emb: np.ndarray, lam: float) -> float:
    """C(X) with the Gaussian kernel (standard pipeline)."""
    return compute_complexity(emb, lam)['C']


def calibrate(dim: int, n_ref: int) -> tuple[float, float]:
    """Calibrate both kernels from one unit-Gaussian reference in R^dim.

    Returns:
        lam: Gaussian-kernel bandwidth (median heuristic)
        c_lin: linear-kernel scale, median of ||Phi_i||^2 (eigenvalues ~ O(1))
    """
    emb_cal = gaussian_reference_embeddings(
        np.zeros(dim), np.eye(dim), SCHEDULE, T_EVAL, n_ref=n_ref)
    lam = calibrate_bandwidth(emb_cal)
    c_lin = np.median(np.sum(emb_cal ** 2, axis=1))
    return lam, c_lin


def n_cluster_embedding(N: int, dim: int) -> np.ndarray:
    """One point per cluster, N unit-Gaussian clusters on a line."""
    means = np.zeros((N, dim))
    means[:, 0] = np.arange(N) * CLUSTER_SPACING
    covs = np.tile(np.eye(dim), (N, 1, 1))
    model = AnalyticScoreModel(means, covs, np.ones(N) / N,
                               noise_schedule=SCHEDULE)
    rng = np.random.default_rng(SEED)
    x = np.array([rng.multivariate_normal(means[k], covs[k]) for k in range(N)])
    return trajectory_embedding(ode_trajectories(x, model, T_EVAL))


def tight_cluster_embedding(N: int, dim: int, sigma: float = 0.1) -> np.ndarray:
    """N points from one tight Gaussian, scored by the unit-Gaussian model."""
    model = AnalyticScoreModel(
        np.zeros((1, dim)), np.tile(np.eye(dim), (1, 1, 1)),
        np.ones(1), noise_schedule=SCHEDULE)
    x = np.random.default_rng(SEED).multivariate_normal(
        np.zeros(dim), sigma ** 2 * np.eye(dim), N)
    return trajectory_embedding(ode_trajectories(x, model, T_EVAL))


def growth_experiment(dim: int, sample_sizes: list[int], n_ref: int,
                      out_name: str) -> None:
    """Sub-experiments 1 and 2: growth rate under both kernels."""
    print("=" * 60)
    print(f"Growth rate, D={dim}")
    print("=" * 60)

    lam, c_lin = calibrate(dim, n_ref)
    print(f"lambda = {lam:.8f}, c = {c_lin:.4f}")

    gauss_clust, lin_clust, gauss_tight, lin_tight = [], [], [], []
    for N in sample_sizes:
        emb = n_cluster_embedding(N, dim)
        gauss_clust.append(C_gaussian(emb, lam))
        lin_clust.append(C_linear(emb, c_lin))
        print(f"N-cluster  N={N:3d}: Gauss={gauss_clust[-1]:.3f}, "
              f"Linear={lin_clust[-1]:.3f}")

        emb = tight_cluster_embedding(N, dim)
        gauss_tight.append(C_gaussian(emb, lam))
        lin_tight.append(C_linear(emb, c_lin))
        print(f"Tight      N={N:3d}: Gauss={gauss_tight[-1]:.3f}, "
              f"Linear={lin_tight[-1]:.3f}")

    Ns = np.array(sample_sizes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, C_clust, C_tight, kernel in [
            (axes[0], gauss_clust, gauss_tight, 'Gaussian'),
            (axes[1], lin_clust, lin_tight, 'Linear')]:
        ax.set_title(f"{kernel} kernel (D={dim})")
        ax.plot(Ns, C_clust, 'o-', label='N-cluster')
        ax.plot(Ns, C_tight, 's-', label='Single cluster (σ=0.1)')
        if kernel == 'Gaussian' and dim == 2:
            ax.plot(Ns, Ns * np.log(2), '--', alpha=0.5, label='N log 2')
            ax.plot(Ns, np.log(1 + Ns), '--', alpha=0.5, label='log(1+N)')
        ax.set_xlabel('N')
        ax.set_ylabel('C(X)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}\n")


def separation_experiment() -> None:
    """Sub-experiment 3: two clusters with growing separation, D=2."""
    print("=" * 60)
    print("Varying separation, D=2")
    print("=" * 60)

    dim, N = 2, 20
    deltas = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30]

    lam, c_lin = calibrate(dim, n_ref=300)

    C_gauss, C_lin = [], []
    for delta in deltas:
        means = np.array([[0, 0], [delta, 0]], dtype=float)
        covs = np.tile(np.eye(dim), (2, 1, 1))
        model = AnalyticScoreModel(means, covs, np.ones(2) / 2,
                                   noise_schedule=SCHEDULE)

        rng = np.random.default_rng(SEED)
        x = np.vstack([
            rng.multivariate_normal(means[0], covs[0], N // 2),
            rng.multivariate_normal(means[1], covs[1], N // 2),
        ])
        emb = trajectory_embedding(ode_trajectories(x, model, T_EVAL))

        C_gauss.append(C_gaussian(emb, lam))
        C_lin.append(C_linear(emb, c_lin))
        print(f"delta={delta:5.1f}: Gauss={C_gauss[-1]:.3f}, Linear={C_lin[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, Cs, kernel in [(axes[0], C_gauss, 'Gaussian'),
                           (axes[1], C_lin, 'Linear')]:
        ax.set_title(f"{kernel} kernel (D=2)")
        ax.plot(deltas, Cs, 'o-')
        ax.set_xlabel('Cluster separation δ')
        ax.set_ylabel('C(X)')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "linear_vs_gaussian_separation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    growth_experiment(dim=2, sample_sizes=[2, 3, 4, 6, 8, 10, 15, 20, 25, 30],
                      n_ref=300, out_name="linear_vs_gaussian_D2.png")
    growth_experiment(dim=100, sample_sizes=[2, 5, 10, 15, 20, 30],
                      n_ref=100, out_name="linear_vs_gaussian_D100.png")
    separation_experiment()


if __name__ == '__main__':
    main()
