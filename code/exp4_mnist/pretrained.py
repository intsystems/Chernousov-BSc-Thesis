"""Pretrained MNIST DDPM (HuggingFace `1aurent/ddpm-mnist`) as a score model.

`DDPMNoiseSchedule` mirrors the discrete linear-beta schedule of the
checkpoint and exposes the continuous-time interface expected by
core/trajectory.py.  `PretrainedMNISTModel` adapts the diffusers UNet to
``score(x, t) -> (N, L, D)``.  `batched_ode_trajectories` integrates the VP
probability-flow ODE with batched Euler steps — per-point adaptive RK45
(core.ode_trajectories) is impractical for a neural score model.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers import DDPMPipeline

IMG_DIM = 784  # flattened 28x28


class DDPMNoiseSchedule:
    """Discrete linear beta schedule of the pretrained model.

    Precomputes alpha_bar[k] for k = 0, ..., T-1 and maps continuous time
    t in [0, 1] to the nearest discrete timestep.
    """

    schedule_type = 'vp'

    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02,
                 T: int = 1000):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = np.linspace(beta_start, beta_end, T)
        self.alpha_bars = np.cumprod(1.0 - betas)

    def t_to_k(self, t: float) -> int:
        """Nearest discrete timestep k for continuous t in [0, 1]."""
        k = int(round(t * (self.T - 1)))
        return max(0, min(self.T - 1, k))

    def alpha_bar(self, t: float) -> float:
        return self.alpha_bars[self.t_to_k(t)]

    def sigma(self, t: float) -> float:
        """sigma(t) = sqrt(1 - alpha_bar(t))."""
        return np.sqrt(1.0 - self.alpha_bar(t))

    def beta(self, t: float) -> float:
        """Continuous-time beta(t) for the VP ODE drift.

        The discrete schedule applies beta_k once per 1/T of unit time, so
        the continuous rate is T times the interpolated discrete value.
        """
        return self.T * (self.beta_start + (self.beta_end - self.beta_start) * t)


class PretrainedMNISTModel:
    """Pretrained DDPM with the ``score(x, t)`` interface of core models.

    Images are flattened 28x28 arrays in [-1, 1].
    """

    def __init__(self, model_id: str = '1aurent/ddpm-mnist', device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        pipe = DDPMPipeline.from_pretrained(model_id)
        self.unet = pipe.unet.to(device).eval()
        self.noise_schedule = DDPMNoiseSchedule(
            beta_start=pipe.scheduler.config.beta_start,
            beta_end=pipe.scheduler.config.beta_end,
            T=pipe.scheduler.config.num_train_timesteps,
        )

    def score(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """s(x, t) = -eps_pred(x, k(t)) / sigma(t).

        Args:
            x: (N, 784) images in [-1, 1]
            t: (L,) times in [0, 1]

        Returns:
            scores: (N, L, 784) array
        """
        x = np.asarray(x, dtype=np.float32)
        t = np.asarray(t, dtype=np.float64)
        N, D = x.shape
        assert D == IMG_DIM, f'Expected D={IMG_DIM} (28x28), got D={D}'

        scores = np.zeros((N, len(t), D), dtype=np.float32)

        with torch.no_grad():
            x_img = torch.from_numpy(x.reshape(N, 1, 28, 28)).to(self.device)

            for j, tj in enumerate(t):
                k = self.noise_schedule.t_to_k(tj)
                timestep = torch.full((N,), k, device=self.device, dtype=torch.long)
                eps_pred = self.unet(x_img, timestep).sample
                score_img = -eps_pred / self.noise_schedule.sigma(tj)
                scores[:, j, :] = score_img.reshape(N, D).cpu().numpy()

        return scores


@torch.no_grad()
def batched_ode_trajectories(x: np.ndarray, model: PretrainedMNISTModel,
                             t_eval: np.ndarray,
                             n_euler_steps: int = 200) -> np.ndarray:
    """VP probability-flow ODE trajectories via batched Euler integration.

    All N points go through the network in one batch per step:
        dz/dt = -1/2 * beta(t) * [z + s(z, t)],   s(z, t) = -eps_pred / sigma(t)

    Args:
        x: (N, 784) images in [-1, 1]
        model: PretrainedMNISTModel
        t_eval: (L,) time points at which to record z(t)
        n_euler_steps: Euler steps over [0, 1]

    Returns:
        trajectories: (N, L, 784) array
    """
    ns = model.noise_schedule
    N, D = x.shape
    L = len(t_eval)

    dt = 1.0 / n_euler_steps
    t_grid = np.linspace(0, 1.0, n_euler_steps + 1)
    # Indices of the Euler steps nearest to each requested time point
    record_indices = [int(np.argmin(np.abs(t_grid - te))) for te in t_eval]

    z = torch.from_numpy(x.reshape(N, 1, 28, 28).astype(np.float32)).to(model.device)
    trajectories = np.zeros((N, L, D), dtype=np.float32)

    for step in range(n_euler_steps + 1):
        if step in record_indices:
            j = record_indices.index(step)
            trajectories[:, j, :] = z.reshape(N, D).cpu().numpy()
        if step == n_euler_steps:
            break

        t = t_grid[step]
        k = ns.t_to_k(t)
        timestep = torch.full((N,), k, device=model.device, dtype=torch.long)
        eps_pred = model.unet(z, timestep).sample
        score = -eps_pred / ns.sigma(t)

        z = z + (-0.5 * ns.beta(t) * (z + score)) * dt

    return trajectories
