"""Shared helpers for the exp3 scripts."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model import AnalyticScoreModel
from core.trajectory import ode_trajectories, trajectory_embedding


def gaussian_reference_embeddings(mean: np.ndarray, cov: np.ndarray,
                                  noise_schedule, t_eval: np.ndarray,
                                  n_ref: int = 300, seed: int = 99) -> np.ndarray:
    """Trajectory embeddings of a reference sample from N(mean, cov).

    The reference sample is scored by the analytic model of its own
    distribution; the result is used to calibrate the kernel bandwidth
    independently of any measured dataset.

    Args:
        mean: (D,) Gaussian mean
        cov: (D, D) Gaussian covariance
        noise_schedule: schedule shared with the experiment's models
        t_eval: (L,) trajectory time grid
        n_ref: reference sample size
        seed: RNG seed for the reference sample

    Returns:
        embeddings: (n_ref, L*D) trajectory embeddings
    """
    model = AnalyticScoreModel(mean[None, :], cov[None, :, :], np.ones(1),
                               noise_schedule=noise_schedule)
    x = np.random.default_rng(seed).multivariate_normal(mean, cov, n_ref)
    traj = ode_trajectories(x, model, t_eval)
    return trajectory_embedding(traj)
