"""Growth rate of C(X) on MNIST with a pretrained diffusion model.

Two curves as functions of the sample size N:
  - single-class: all samples are the digit 0
  - multi-class: samples cycle through the digits 0, 1, ..., 9

Trajectories come from batched Euler integration of the PF-ODE
(see exp4_mnist/pretrained.py); time is weighted by a Beta(2, 2) pdf.
Output: results/mnist_growth.png

Usage:
    python run_growth.py            # full run (slow on CPU)
    python run_growth.py --quick    # reduced smoke test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp4_mnist.pretrained import PretrainedMNISTModel, batched_ode_trajectories
from core.trajectory import trajectory_embedding, beta_weights
from core.complexity import compute_complexity, calibrate_bandwidth

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
DATA_DIR = Path(__file__).resolve().parent / 'data'


def load_mnist_by_digit() -> dict[int, np.ndarray]:
    """MNIST train images grouped by digit: digit -> (M, 784) array in [-1, 1]."""
    ds = datasets.MNIST(str(DATA_DIR), train=True, download=True)
    images = ds.data.numpy().astype(np.float32) / 255.0 * 2.0 - 1.0
    labels = ds.targets.numpy()
    return {d: images[labels == d].reshape(-1, 784) for d in range(10)}


def build_cycling_sample(by_digit: dict[int, np.ndarray], N: int) -> np.ndarray:
    """Sample of size N cycling through the digits 0, 1, ..., 9, 0, 1, ..."""
    counts = {d: 0 for d in range(10)}
    samples = []
    for i in range(N):
        d = i % 10
        samples.append(by_digit[d][counts[d]])
        counts[d] += 1
    return np.array(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick', action='store_true',
                        help='reduced sizes for a smoke test')
    args = parser.parse_args()

    n_euler = 50 if args.quick else 200
    N_values = [5, 10] if args.quick else [5, 10, 15, 20, 30, 40, 50]

    RESULTS_DIR.mkdir(exist_ok=True)

    print('Loading MNIST...')
    by_digit = load_mnist_by_digit()

    print('Loading pretrained model...')
    model = PretrainedMNISTModel()

    L = 10
    t_eval = np.linspace(0.05, 0.95, L)
    weights = beta_weights(t_eval, alpha=2.0, beta=2.0)

    # Calibrate lambda on a held-out reference sample (30 zeros)
    print('Calibrating lambda...')
    ref_data = by_digit[0][200:230]
    ref_trajs = batched_ode_trajectories(ref_data, model, t_eval, n_euler_steps=n_euler)
    ref_emb = trajectory_embedding(ref_trajs, weights)
    lambda_ = calibrate_bandwidth(ref_emb)
    print(f'  lambda = {lambda_:.4f}')

    C_single, C_multi = [], []

    for N in N_values:
        print(f'--- N = {N} ---')

        x_single = by_digit[0][:N]
        trajs = batched_ode_trajectories(x_single, model, t_eval, n_euler_steps=n_euler)
        res = compute_complexity(trajectory_embedding(trajs, weights), lambda_)
        C_single.append(res['C'])
        print(f'  C_single = {res["C"]:.4f}')

        x_multi = build_cycling_sample(by_digit, N)
        trajs = batched_ode_trajectories(x_multi, model, t_eval, n_euler_steps=n_euler)
        res = compute_complexity(trajectory_embedding(trajs, weights), lambda_)
        C_multi.append(res['C'])
        print(f'  C_multi  = {res["C"]:.4f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, C_single, 'o-', color='steelblue', lw=2,
            label='Single class (digit 0)')
    ax.plot(N_values, C_multi, 's-', color='firebrick', lw=2,
            label='Multi-class (cycling 0–9)')
    ax.set_xlabel('Sample size N', fontsize=12)
    ax.set_ylabel('C(X)', fontsize=12)
    ax.set_title('Dataset complexity growth rate on MNIST', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    suffix = '_quick' if args.quick else ''
    path = RESULTS_DIR / f'mnist_growth{suffix}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\nSaved: {path}')

    print(f'\n{"N":>5} {"C_single":>10} {"C_multi":>10} {"ratio":>8}')
    for i, N in enumerate(N_values):
        ratio = C_multi[i] / C_single[i] if C_single[i] > 0 else float('inf')
        print(f'{N:>5} {C_single[i]:>10.4f} {C_multi[i]:>10.4f} {ratio:>8.2f}')


if __name__ == '__main__':
    main()
