"""Complexity of a GMM sample as a function of cluster separation.

Lambda is calibrated from a single-cluster reference so it captures the
within-cluster scale and is not washed out by between-cluster distances.
Output: results/varying_separation.png
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

N_CLUSTERS = 3
DIM = 2
N_POINTS = 30
SCHEDULE = VPNoiseSchedule()
T_EVAL = np.linspace(0.01, 0.99, 20)
SEPARATIONS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]


def make_gmm_means(K: int, delta: float) -> np.ndarray:
    """K means equally spaced on a circle of radius delta (D=2)."""
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return delta * np.column_stack([np.cos(angles), np.sin(angles)])


def sample_gmm_roundrobin(means, covs, N, seed=None):
    """Sample N points cycling through the clusters deterministically."""
    rng = np.random.default_rng(seed)
    K = len(means)
    x = np.zeros((N, means.shape[1]))
    for i in range(N):
        x[i] = rng.multivariate_normal(means[i % K], covs[i % K])
    return x


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Single-cluster calibration: captures within-cluster scale,
    # independent of the separation being varied below.
    emb_cal = gaussian_reference_embeddings(
        np.zeros(DIM), np.eye(DIM), SCHEDULE, T_EVAL)
    lam = calibrate_bandwidth(emb_cal)
    print(f"Lambda (single-cluster calibration): {lam:.6f}")

    C_singles, C_mixtures = [], []

    for delta in SEPARATIONS:
        means = make_gmm_means(N_CLUSTERS, delta)
        covs = np.tile(np.eye(DIM), (N_CLUSTERS, 1, 1))
        model = AnalyticScoreModel(means, covs, np.ones(N_CLUSTERS) / N_CLUSTERS,
                                   noise_schedule=SCHEDULE)

        # Single cluster
        x_s = np.random.default_rng(SEED).multivariate_normal(
            means[0], covs[0], N_POINTS)
        emb_s = trajectory_embedding(ode_trajectories(x_s, model, T_EVAL))
        res_s = compute_complexity(emb_s, lam)

        # Mixture (round-robin over clusters)
        x_m = sample_gmm_roundrobin(means, covs, N_POINTS, seed=SEED)
        emb_m = trajectory_embedding(ode_trajectories(x_m, model, T_EVAL))
        res_m = compute_complexity(emb_m, lam)

        C_singles.append(res_s['C'])
        C_mixtures.append(res_m['C'])
        print(f"delta={delta:5.1f}: C_single={res_s['C']:.3f}, "
              f"C_mix={res_m['C']:.3f}, ratio={res_m['C'] / res_s['C']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SEPARATIONS, C_mixtures, 's-', label=f'Mixture ({N_CLUSTERS} clusters)')
    ax.set_xlabel('Separation $\\delta$')
    ax.set_ylabel('$C(X)$')
    ax.set_title('Complexity vs cluster separation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "varying_separation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
