"""Probability-flow ODE trajectories and the trajectory embedding.

Pipeline: data points (N, D) -> PF-ODE trajectories (N, L, D)
-> weighted flattened embedding (N, L*D), whose Euclidean distance is a
quadrature approximation of D^2(x_i, x_j) = int ||z_i(t) - z_j(t)||^2 w(t) dt.

PF-ODE drift (z(0) = x):
    VE: dz/dt = -ln(sigma_max / sigma_min) * sigma(t)^2 * s(z, t)
    VP: dz/dt = -1/2 * beta(t) * [z + s(z, t)]
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import beta as beta_dist


def ode_trajectories(x: np.ndarray, model, t_eval: np.ndarray,
                     rtol: float = 1e-5, atol: float = 1e-7) -> np.ndarray:
    """Integrate the probability-flow ODE from each data point.

    The drift is chosen by the model's noise schedule (`schedule_type`
    attribute, 've' or 'vp').

    Args:
        x: (N, D) data points
        model: score model with .score(x, t) -> (N, L, D) and .noise_schedule
        t_eval: (L,) time points at which to record z(t)
        rtol, atol: tolerances for the RK45 solver

    Returns:
        trajectories: (N, L, D) array of z_{x_i}(t_l)

    Raises:
        RuntimeError: if the solver fails for some data point
    """
    N, D = x.shape
    ns = model.noise_schedule
    is_vp = getattr(ns, 'schedule_type', 've') == 'vp'
    t_eval = np.sort(t_eval)
    L = len(t_eval)

    if not is_vp:
        log_ratio = np.log(ns.sigma_max / ns.sigma_min)

    trajectories = np.zeros((N, L, D))

    for i in range(N):
        def rhs(t, z):
            s = model.score(z.reshape(1, D), np.array([t]))[0, 0, :]
            if is_vp:
                return -0.5 * ns.beta(t) * (z + s)
            return -log_ratio * ns.sigma(t) ** 2 * s

        sol = solve_ivp(rhs, [0, t_eval[-1]], x[i],
                        t_eval=t_eval, method='RK45', rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'PF-ODE failed for point {i}: {sol.message}')

        trajectories[i] = sol.y.T

    return trajectories


def beta_weights(t_eval: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Beta(alpha, beta) pdf weights at the given time points, summing to 1.

    Turns the trajectory distance into an expectation
    D^2(x_i, x_j) = E_{t ~ Beta(alpha, beta)} ||z_i(t) - z_j(t)||^2.

    Args:
        t_eval: (L,) time points in (0, 1)
        alpha, beta: Beta distribution shape parameters (> 0)

    Returns:
        weights: (L,) normalized weights (uniform if the pdf vanishes
        on the whole grid)
    """
    w = beta_dist.pdf(t_eval, alpha, beta)
    w_sum = w.sum()
    if w_sum == 0:
        return np.ones(len(t_eval)) / len(t_eval)
    return w / w_sum


def trajectory_embedding(trajectories: np.ndarray,
                         weights: np.ndarray | None = None) -> np.ndarray:
    """Flatten trajectories into embeddings with sqrt-weighted time slices.

    f(x) = (sqrt(w_1) z_x(t_1), ..., sqrt(w_L) z_x(t_L)), so that
    ||f(x_i) - f(x_j)||^2 = sum_l w_l ||z_i(t_l) - z_j(t_l)||^2.

    Args:
        trajectories: (N, L, D) PF-ODE trajectories
        weights: (L,) time weights (normalized internally), or None for uniform

    Returns:
        embeddings: (N, L*D) array
    """
    N, L, D = trajectories.shape
    if weights is None:
        weights = np.ones(L)
    weights = weights / weights.sum()

    weighted = trajectories * np.sqrt(weights)[np.newaxis, :, np.newaxis]
    return weighted.reshape(N, L * D)
