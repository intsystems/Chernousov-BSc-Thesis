"""Complexity vs N for different intrinsic dimensions.

Data lives on d-dimensional subspaces of R^10 (d = 1, 2, 3, 5).  Each
dataset is one Gaussian with diagonal covariance
    Sigma = diag(sigma^2, ..., sigma^2, eps, ..., eps),
            |---- d active ----|  |- D-d near-zero -|
and the analytic score model matches the data distribution in each case.
Output: results/manifold_dimension.png
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

DIM_AMBIENT = 10
SIGMA_ACTIVE = 1.0   # variance per active direction
EPS_FLOOR = 1e-4     # variance in inactive directions (avoids singular cov)
INTRINSIC_DIMS = [1, 2, 3, 5]
SAMPLE_SIZES = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50]
DIM_REFERENCE = 3    # intrinsic dim of the calibration reference
SCHEDULE = VPNoiseSchedule()
T_EVAL = np.linspace(0.01, 0.99, 20)


def subspace_covariance(d: int) -> np.ndarray:
    """Diagonal covariance with d active directions in R^DIM_AMBIENT."""
    cov = EPS_FLOOR * np.eye(DIM_AMBIENT)
    cov[np.arange(d), np.arange(d)] = SIGMA_ACTIVE
    return cov


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    emb_cal = gaussian_reference_embeddings(
        np.zeros(DIM_AMBIENT), subspace_covariance(DIM_REFERENCE),
        SCHEDULE, T_EVAL)
    lam = calibrate_bandwidth(emb_cal)
    print(f"Lambda (calibrated from d={DIM_REFERENCE} reference): {lam:.6f}\n")

    results = {}
    for d in INTRINSIC_DIMS:
        cov = subspace_covariance(d)
        mean = np.zeros(DIM_AMBIENT)
        model = AnalyticScoreModel(mean[None, :], cov[None, :, :], np.ones(1),
                                   noise_schedule=SCHEDULE)

        Cs = []
        for N in SAMPLE_SIZES:
            x = np.random.default_rng(SEED).multivariate_normal(mean, cov, N)
            emb = trajectory_embedding(ode_trajectories(x, model, T_EVAL))
            Cs.append(compute_complexity(emb, lam)['C'])

        results[d] = Cs
        print(f"d={d}: {['%.2f' % c for c in Cs]}")

    Ns = np.array(SAMPLE_SIZES)
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in INTRINSIC_DIMS:
        ax.plot(Ns, results[d], 'o-', label=f'd = {d}')
    ax.plot(Ns, np.log(1 + Ns), '--', color='gray', alpha=0.5,
            label='$\\log(1+N)$ (identical)')
    ax.set_xlabel('$N$')
    ax.set_ylabel('$C(X)$')
    ax.set_title(
        f'Complexity vs $N$ for different intrinsic dimensions '
        f'($D = {DIM_AMBIENT}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "manifold_dimension.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
