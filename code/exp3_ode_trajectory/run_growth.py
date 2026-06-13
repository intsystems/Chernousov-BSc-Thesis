"""Growth rate of C(X): linear vs logarithmic.

Two regimes with the trajectory embedding and analytic VP scores:
  - new cluster per point (well-separated):  C ~ N log 2
  - single cluster (tight variance):         C ~ log(1 + N)

Lambda is calibrated once from a unit-Gaussian reference sample.
Output: results/growth_rate.png (the thesis copy lives in figures/).
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

DIM = 2
SCHEDULE = VPNoiseSchedule()
T_EVAL = np.linspace(0.01, 0.99, 20)
SAMPLE_SIZES = [2, 3, 4, 6, 8, 10, 15, 20, 25, 30]
CLUSTER_SPACING = 20.0  # distance between adjacent cluster means
SINGLE_CLUSTER_SIGMAS = [0.1, 1.0]


def n_cluster_complexities(lam: float) -> list[float]:
    """One point per cluster, N clusters on a line: C should grow ~ N log 2."""
    Cs = []
    for N in SAMPLE_SIZES:
        means = np.zeros((N, DIM))
        means[:, 0] = np.arange(N) * CLUSTER_SPACING
        covs = np.tile(np.eye(DIM), (N, 1, 1))
        model = AnalyticScoreModel(means, covs, np.ones(N) / N,
                                   noise_schedule=SCHEDULE)

        rng = np.random.default_rng(SEED)
        x = np.array([rng.multivariate_normal(means[k], covs[k]) for k in range(N)])
        traj = ode_trajectories(x, model, T_EVAL)
        emb = trajectory_embedding(traj)
        res = compute_complexity(emb, lam)
        Cs.append(res['C'])
        print(f"N-cluster  N={N:3d}: C={res['C']:.3f}, theory={N * np.log(2):.3f}")
    return Cs


def single_cluster_complexities(lam: float, sigma: float) -> list[float]:
    """All N points from one tight cluster of a 3-component GMM: C ~ log(1+N)."""
    K = 3
    means = np.array([[5., 0.], [-2.5, 4.33], [-2.5, -4.33]])
    covs = np.tile(np.eye(DIM), (K, 1, 1))
    model = AnalyticScoreModel(means, covs, np.ones(K) / K,
                               noise_schedule=SCHEDULE)

    Cs = []
    for N in SAMPLE_SIZES:
        x = np.random.default_rng(SEED).multivariate_normal(
            means[0], sigma ** 2 * np.eye(DIM), N)
        traj = ode_trajectories(x, model, T_EVAL)
        emb = trajectory_embedding(traj)
        res = compute_complexity(emb, lam)
        Cs.append(res['C'])
        print(f"Single(σ={sigma})  N={N:3d}: C={res['C']:.3f}")
    return Cs


def plot(C_nclust, C_single):
    Ns = np.array(SAMPLE_SIZES)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ns, C_nclust, 'o-', label='New cluster per point')
    ax.plot(Ns, C_single[1.0], 's-', label='Single cluster ($\\sigma = 1$)')
    ax.plot(Ns, C_single[0.1], '^-', label='Single cluster ($\\sigma = 0.1$)')
    ax.plot(Ns, Ns * np.log(2), '--', color='C0', alpha=0.5, label='$N \\log 2$')
    ax.plot(Ns, np.log(1 + Ns), '--', color='C2', alpha=0.5, label='$\\log(1+N)$')
    ax.set_xlabel('$N$')
    ax.set_ylabel('$C(X)$')
    ax.set_title('Complexity growth rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "growth_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    emb_cal = gaussian_reference_embeddings(
        np.zeros(DIM), np.eye(DIM), SCHEDULE, T_EVAL)
    lam = calibrate_bandwidth(emb_cal)
    print(f"Lambda: {lam:.6f}")

    C_nclust = n_cluster_complexities(lam)
    C_single = {sig: single_cluster_complexities(lam, sig)
                for sig in SINGLE_CLUSTER_SIGMAS}

    plot(C_nclust, C_single)


if __name__ == '__main__':
    main()
