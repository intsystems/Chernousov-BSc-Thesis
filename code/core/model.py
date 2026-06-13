"""Score-function models.

`AnalyticScoreModel` gives the exact score of a Gaussian mixture under a VE
or VP noise schedule, so the complexity pipeline can be validated with zero
approximation error.  Neural models (exp4) implement the same
``score(x, t) -> (N, L, D)`` interface.
"""

from __future__ import annotations

import numpy as np


class NoiseSchedule:
    """Variance-exploding (VE) schedule.

    sigma(t) = sigma_min * (sigma_max / sigma_min)^t,  t in [0, 1].
    Forward kernel: q(x_t | x_0) = N(x_0, sigma(t)^2 I).
    """

    schedule_type = 've'

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t) -> np.ndarray:
        """sigma(t) for scalar or array t."""
        t = np.asarray(t, dtype=np.float64)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def uniform_grid(self, L: int) -> np.ndarray:
        """L time points uniformly spaced in [0, 1] (log-spaced sigmas)."""
        return np.linspace(0, 1, L)

    def sigma_grid(self, L: int) -> np.ndarray:
        """sigma values at L uniformly spaced time points."""
        return self.sigma(self.uniform_grid(L))


class VPNoiseSchedule:
    """Variance-preserving (VP) schedule.

    beta(t) = beta_min + (beta_max - beta_min) * t
    alpha_bar(t) = exp(-beta_min * t - (beta_max - beta_min) * t^2 / 2)
    Forward kernel: q(x_t | x_0) = N(sqrt(alpha_bar) x_0, (1 - alpha_bar) I),
    so x_0 is recovered at t=0 and x_1 ~ N(0, I).
    """

    schedule_type = 'vp'

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t) -> np.ndarray:
        """beta(t) for scalar or array t."""
        t = np.asarray(t, dtype=np.float64)
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t) -> np.ndarray:
        """alpha_bar(t) = exp(-int_0^t beta)."""
        t = np.asarray(t, dtype=np.float64)
        return np.exp(-self.beta_min * t - 0.5 * (self.beta_max - self.beta_min) * t ** 2)

    def sigma(self, t) -> np.ndarray:
        """sigma(t) = sqrt(1 - alpha_bar(t))."""
        return np.sqrt(1.0 - self.alpha_bar(t))

    def signal_rate(self, t) -> np.ndarray:
        """sqrt(alpha_bar(t)) — scaling of the signal component."""
        return np.sqrt(self.alpha_bar(t))

    def uniform_grid(self, L: int) -> np.ndarray:
        return np.linspace(0, 1, L)

    def sigma_grid(self, L: int) -> np.ndarray:
        return self.sigma(self.uniform_grid(L))


class AnalyticScoreModel:
    """Exact score s(x, t) = grad_x log p_t(x) for a Gaussian mixture.

    For p(x) = sum_k pi_k N(x; mu_k, Sigma_k) the noisy marginal stays a
    mixture of Gaussians under both schedules:

        VE: p_t(x) = sum_k pi_k N(x; mu_k,            Sigma_k + sigma(t)^2 I)
        VP: p_t(x) = sum_k pi_k N(x; sqrt(abar) mu_k, abar Sigma_k + (1 - abar) I)

    and the score is the posterior-weighted pull toward the (scaled) means.
    """

    def __init__(self, means, covariances=None, weights=None,
                 noise_schedule=None):
        """
        Args:
            means: (K, D) component means
            covariances: (K, D, D) covariance matrices, (K,) scalar variances
                (isotropic), or None for unit covariance
            weights: (K,) mixture weights, or None for uniform
            noise_schedule: NoiseSchedule or VPNoiseSchedule, default VE
        """
        self.means = np.asarray(means, dtype=np.float64)
        self.n_components, self.dim = self.means.shape

        if covariances is None:
            self.covs = np.tile(np.eye(self.dim), (self.n_components, 1, 1))
        else:
            covariances = np.asarray(covariances, dtype=np.float64)
            if covariances.ndim == 1:
                self.covs = covariances[:, None, None] * np.eye(self.dim)
            else:
                self.covs = covariances

        if weights is None:
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights /= self.weights.sum()

        self.noise_schedule = noise_schedule or NoiseSchedule()

    def score(self, x, t) -> np.ndarray:
        """Evaluate the score at every data point and noise level.

        Args:
            x: (N, D) data points
            t: (L,) noise levels in [0, 1]

        Returns:
            scores: (N, L, D) array, scores[i, l] = s(x_i, t_l)
        """
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        N, D = x.shape
        L = len(t)
        K = self.n_components
        is_vp = getattr(self.noise_schedule, 'schedule_type', 've') == 'vp'

        scores = np.zeros((N, L, D))

        for l in range(L):
            if is_vp:
                abar = self.noise_schedule.alpha_bar(t[l])
                means_l = np.sqrt(abar) * self.means
                covs_l = abar * self.covs + (1 - abar) * np.eye(D)
            else:
                sig2 = self.noise_schedule.sigma(t[l]) ** 2
                means_l = self.means
                covs_l = self.covs + sig2 * np.eye(D)

            cov_invs = np.linalg.inv(covs_l)                     # (K, D, D)
            _, logdets = np.linalg.slogdet(covs_l)               # (K,)
            diffs = means_l[None, :, :] - x[:, None, :]          # (N, K, D)

            # Component log-densities, then posterior weights via log-sum-exp
            mahal = np.einsum('nkd,kde,nke->nk', diffs, cov_invs, diffs)
            log_probs = (np.log(self.weights)[None, :]
                         - 0.5 * D * np.log(2 * np.pi)
                         - 0.5 * logdets[None, :]
                         - 0.5 * mahal)                          # (N, K)
            w = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)

            # s(x, t_l) = sum_k w_k * Sigma_{k,l}^{-1} (mu_{k,l} - x)
            pulls = np.einsum('kde,nke->nkd', cov_invs, diffs)   # (N, K, D)
            scores[:, l, :] = np.einsum('nk,nkd->nd', w, pulls)

        return scores.astype(np.float32)
