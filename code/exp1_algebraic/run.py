"""
Experiment 1: Algebraic property verification.

Tests every proven property of C(X) = log det(I + K) directly on
synthetic score embeddings. No model needed — we construct score
vectors / kernel matrices by hand and verify the math.

Properties tested:
    1. Permutation invariance
    2. Bounded monotonicity: 0 < gain <= log 2
    3. Sub-additivity (Fischer's inequality)
    4. Submodularity (diminishing returns)
    5. Asymptotic cases: identical points, maximally diverse, M clusters
    6. Kernel PSD (eigenvalues >= 0)
    7. CND of D^2 (sum c_i c_j D^2_{ij} <= 0 for sum c_i = 0)
    8. Lipschitz stability under embedding perturbation

Usage:
    python run.py

Exit code is 0 iff all checks pass.
"""

import sys
from pathlib import Path

import numpy as np

# Allow imports from experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.complexity import (
    score_embedding,
    pairwise_distance,
    calibrate_bandwidth,
    median_heuristic,
    gaussian_kernel,
    laplacian_kernel,
    complexity,
    compute_complexity,
)


# ============================================================
# Helpers
# ============================================================

def random_score_vectors(N, L, D, seed=None):
    """Generate random score vectors (N, L, D)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, L, D))


def complexity_from_K(K):
    """Shorthand: returns scalar C only."""
    C, _ = complexity(K)
    return C


def build_kernel(score_vectors, lambda_=None, kernel_fn=gaussian_kernel):
    """Score vectors -> kernel matrix (convenience)."""
    emb = score_embedding(score_vectors)
    D = pairwise_distance(emb)
    if lambda_ is None:
        lambda_ = median_heuristic(D)
    return kernel_fn(D, lambda_), lambda_


# ============================================================
# Tests
# ============================================================

class Results:
    """Collect pass/fail results."""

    def __init__(self):
        self.tests = []

    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        self.tests.append((name, status, detail))
        mark = "✓" if condition else "✗"
        print(f"  [{mark}] {name}")
        if detail:
            print(f"      {detail}")
        return condition

    def summary(self):
        passed = sum(1 for _, s, _ in self.tests if s == "PASS")
        total = len(self.tests)
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed")
        if passed < total:
            print("FAILED:")
            for name, status, detail in self.tests:
                if status == "FAIL":
                    print(f"  - {name}: {detail}")
        print(f"{'='*60}")
        return passed == total


def test_permutation_invariance(R, n_trials=5):
    """C(X) is unchanged under any permutation of rows."""
    print("\n--- Permutation Invariance ---")
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        N = rng.integers(5, 20)
        sv = random_score_vectors(N, 4, 3, seed=100 + trial)
        K_orig, lam = build_kernel(sv)
        C_orig = complexity_from_K(K_orig)

        perm = rng.permutation(N)
        sv_perm = sv[perm]
        K_perm, _ = build_kernel(sv_perm, lambda_=lam)
        C_perm = complexity_from_K(K_perm)

        R.check(
            f"permutation_invariance (N={N}, trial {trial+1})",
            np.isclose(C_orig, C_perm, atol=1e-10),
            f"C_orig={C_orig:.10f}, C_perm={C_perm:.10f}",
        )


def test_bounded_monotonicity(R, n_trials=10):
    """Adding any point: 0 < gain <= log 2."""
    print("\n--- Bounded Monotonicity ---")
    rng = np.random.default_rng(0)
    log2 = np.log(2)

    for trial in range(n_trials):
        N = rng.integers(3, 15)
        L, D = 4, 3
        sv_full = random_score_vectors(N + 1, L, D, seed=200 + trial)
        sv_base = sv_full[:N]

        # Use fixed lambda so adding a point doesn't change bandwidth
        _, lam = build_kernel(sv_full)

        K_base, _ = build_kernel(sv_base, lambda_=lam)
        K_full, _ = build_kernel(sv_full, lambda_=lam)

        C_base = complexity_from_K(K_base)
        C_full = complexity_from_K(K_full)
        gain = C_full - C_base

        R.check(
            f"monotonicity_positive (N={N}, trial {trial+1})",
            gain > 0,
            f"gain={gain:.10f}",
        )
        R.check(
            f"monotonicity_upper_bound (N={N}, trial {trial+1})",
            gain <= log2 + 1e-12,
            f"gain={gain:.10f}, log2={log2:.10f}",
        )


def test_monotonicity_extremes(R):
    """
    Extreme cases:
    - Identical point: gain -> small (approaches log((N+2)/(N+1)))
    - Maximally far point: gain -> log 2
    """
    print("\n--- Monotonicity Extreme Cases ---")
    log2 = np.log(2)

    # Identical point added to N identical points
    N = 10
    L, D = 3, 2
    sv_base = np.ones((N, L, D))
    sv_full = np.ones((N + 1, L, D))
    lam = 1.0
    K_base, _ = build_kernel(sv_base, lambda_=lam)
    K_full, _ = build_kernel(sv_full, lambda_=lam)
    gain_ident = complexity_from_K(K_full) - complexity_from_K(K_base)
    expected = np.log(1 + N + 1) - np.log(1 + N)

    R.check(
        "identical_point_gain",
        np.isclose(gain_ident, expected, atol=1e-10),
        f"gain={gain_ident:.10f}, expected log((N+2)/(N+1))={expected:.10f}",
    )

    # Maximally far point: make one point have huge embedding distance
    sv_base2 = np.zeros((N, L, D))
    sv_far = np.zeros((1, L, D))
    sv_far[0, :, :] = 1e6  # very far in embedding space
    sv_full2 = np.concatenate([sv_base2, sv_far], axis=0)
    K_base2, _ = build_kernel(sv_base2, lambda_=lam)
    K_full2, _ = build_kernel(sv_full2, lambda_=lam)
    gain_far = complexity_from_K(K_full2) - complexity_from_K(K_base2)

    R.check(
        "maximally_far_point_gain_approaches_log2",
        np.isclose(gain_far, log2, atol=1e-6),
        f"gain={gain_far:.10f}, log2={log2:.10f}",
    )


def test_sub_additivity(R, n_trials=10):
    """C(A ∪ B) <= C(A) + C(B) for disjoint A, B."""
    print("\n--- Sub-additivity (Fischer's Inequality) ---")
    rng = np.random.default_rng(1)

    for trial in range(n_trials):
        Na = rng.integers(3, 10)
        Nb = rng.integers(3, 10)
        L, D = 4, 3
        sv_all = random_score_vectors(Na + Nb, L, D, seed=300 + trial)
        sv_a = sv_all[:Na]
        sv_b = sv_all[Na:]

        # Fixed lambda for all three computations
        _, lam = build_kernel(sv_all)

        K_a, _ = build_kernel(sv_a, lambda_=lam)
        K_b, _ = build_kernel(sv_b, lambda_=lam)
        K_all, _ = build_kernel(sv_all, lambda_=lam)

        C_a = complexity_from_K(K_a)
        C_b = complexity_from_K(K_b)
        C_all = complexity_from_K(K_all)

        gap = (C_a + C_b) - C_all

        R.check(
            f"sub_additivity (Na={Na}, Nb={Nb}, trial {trial+1})",
            gap >= -1e-10,
            f"C(A)+C(B)={C_a+C_b:.6f}, C(A∪B)={C_all:.6f}, gap={gap:.6f}",
        )


def test_submodularity(R, n_trials=10):
    """
    For X ⊆ X' and x ∉ X':
    gain(x | X) >= gain(x | X').
    """
    print("\n--- Submodularity (Diminishing Returns) ---")
    rng = np.random.default_rng(2)

    for trial in range(n_trials):
        N_prime = rng.integers(5, 15)
        N_sub = rng.integers(2, N_prime)
        L, D = 4, 3

        # X' has N_prime points, X is a subset, x is one extra point
        sv_pool = random_score_vectors(N_prime + 1, L, D, seed=400 + trial)
        sv_x = sv_pool[-1:]  # the point to add

        # X ⊆ X'
        indices_prime = np.arange(N_prime)
        indices_sub = rng.choice(N_prime, size=N_sub, replace=False)
        indices_sub.sort()

        sv_X = sv_pool[indices_sub]
        sv_Xprime = sv_pool[indices_prime]

        # Fixed lambda
        sv_all = np.concatenate([sv_Xprime, sv_x], axis=0)
        _, lam = build_kernel(sv_all)

        # Gain when adding x to X
        sv_X_plus = np.concatenate([sv_X, sv_x], axis=0)
        K_X, _ = build_kernel(sv_X, lambda_=lam)
        K_X_plus, _ = build_kernel(sv_X_plus, lambda_=lam)
        gain_small = complexity_from_K(K_X_plus) - complexity_from_K(K_X)

        # Gain when adding x to X'
        sv_Xp_plus = np.concatenate([sv_Xprime, sv_x], axis=0)
        K_Xp, _ = build_kernel(sv_Xprime, lambda_=lam)
        K_Xp_plus, _ = build_kernel(sv_Xp_plus, lambda_=lam)
        gain_large = complexity_from_K(K_Xp_plus) - complexity_from_K(K_Xp)

        R.check(
            f"submodularity (|X|={N_sub}, |X'|={N_prime}, trial {trial+1})",
            gain_small >= gain_large - 1e-10,
            f"gain(x|X)={gain_small:.6f}, gain(x|X')={gain_large:.6f}",
        )


def test_identical_points(R):
    """N identical points: C = log(1 + N)."""
    print("\n--- Asymptotic: Identical Points ---")

    for N in [1, 2, 5, 10, 50, 100]:
        L, D = 3, 2
        sv = np.ones((N, L, D)) * 0.5  # any constant
        K, _ = build_kernel(sv, lambda_=1.0)
        C = complexity_from_K(K)
        expected = np.log(1 + N)

        R.check(
            f"identical_points (N={N})",
            np.isclose(C, expected, atol=1e-10),
            f"C={C:.10f}, expected={expected:.10f}",
        )


def test_maximally_diverse(R):
    """N maximally diverse points (K -> I): C -> N log 2."""
    print("\n--- Asymptotic: Maximally Diverse ---")

    for N in [2, 5, 10, 20]:
        L, D = 3, 2
        # Points very far apart in embedding space
        sv = np.zeros((N, L, D))
        for i in range(N):
            sv[i, :, :] = i * 1e6

        K, _ = build_kernel(sv, lambda_=1.0)
        C = complexity_from_K(K)
        expected = N * np.log(2)

        R.check(
            f"maximally_diverse (N={N})",
            np.isclose(C, expected, atol=1e-6),
            f"C={C:.6f}, expected={expected:.6f}",
        )


def test_m_clusters(R):
    """M clusters of m identical points: C = M * log(1 + m)."""
    print("\n--- Asymptotic: M Clusters ---")

    for M, m in [(2, 5), (3, 4), (5, 10), (10, 2)]:
        N = M * m
        L, D = 3, 2

        sv = np.zeros((N, L, D))
        for c in range(M):
            # Each cluster has a unique embedding, members identical
            sv[c * m:(c + 1) * m, :, :] = c * 1e6

        K, _ = build_kernel(sv, lambda_=1.0)
        C = complexity_from_K(K)
        expected = M * np.log(1 + m)

        R.check(
            f"m_clusters (M={M}, m={m})",
            np.isclose(C, expected, atol=1e-6),
            f"C={C:.6f}, expected={expected:.6f}",
        )


def test_kernel_psd(R, n_trials=5):
    """Kernel matrix eigenvalues are all non-negative."""
    print("\n--- Kernel PSD ---")

    for trial in range(n_trials):
        sv = random_score_vectors(20, 4, 3, seed=500 + trial)

        for name, kfn in [("gaussian", gaussian_kernel),
                          ("laplacian", laplacian_kernel)]:
            K, _ = build_kernel(sv, kernel_fn=kfn)
            eigs = np.linalg.eigvalsh(K)
            min_eig = eigs.min()

            R.check(
                f"kernel_psd_{name} (trial {trial+1})",
                min_eig >= -1e-10,
                f"min eigenvalue = {min_eig:.2e}",
            )


def test_cnd_d_squared(R, n_trials=5):
    """
    D^2 is CND: for any c with sum(c)=0, sum_ij c_i c_j D^2_{ij} <= 0.
    Equivalent to: sum = -2 ||sum_i c_i a_i||^2 <= 0.
    """
    print("\n--- CND of D^2 ---")
    rng = np.random.default_rng(3)

    for trial in range(n_trials):
        N = rng.integers(5, 20)
        sv = random_score_vectors(N, 4, 3, seed=600 + trial)
        emb = score_embedding(sv)
        D = pairwise_distance(emb)
        D_sq = D ** 2

        # Random coefficients with sum = 0
        c = rng.standard_normal(N)
        c -= c.mean()  # enforce sum = 0

        quadratic = c @ D_sq @ c

        R.check(
            f"cnd_d_squared (N={N}, trial {trial+1})",
            quadratic <= 1e-10,
            f"sum c_i c_j D^2_ij = {quadratic:.2e}",
        )


def test_lipschitz_stability(R, n_trials=5):
    """
    Master perturbation bound:
    |C' - C| <= 4 * N^{3/2} * lambda * eta * (D_max + eta)
    where eta = max_i ||a'_i - a_i||.
    """
    print("\n--- Lipschitz Stability ---")
    rng = np.random.default_rng(4)

    for trial in range(n_trials):
        N = rng.integers(5, 15)
        L, D = 4, 3
        sv = random_score_vectors(N, L, D, seed=700 + trial)

        emb = score_embedding(sv)
        D_mat = pairwise_distance(emb)
        lam = median_heuristic(D_mat)

        K = gaussian_kernel(D_mat, lam)
        C_orig = complexity_from_K(K)
        D_max = D_mat.max()

        # Perturb embeddings
        eta_scale = rng.uniform(0.01, 0.5)
        perturbation = rng.standard_normal(sv.shape) * eta_scale
        sv_pert = sv + perturbation

        emb_pert = score_embedding(sv_pert)
        eta = np.max(np.linalg.norm(emb_pert - emb, axis=1))

        D_pert = pairwise_distance(emb_pert)
        K_pert = gaussian_kernel(D_pert, lam)
        C_pert = complexity_from_K(K_pert)

        actual_diff = abs(C_pert - C_orig)
        bound = 4 * N**1.5 * lam * eta * (D_max + eta)

        R.check(
            f"lipschitz_bound (N={N}, trial {trial+1})",
            actual_diff <= bound + 1e-10,
            f"|ΔC|={actual_diff:.6f}, bound={bound:.6f}",
        )


def test_compute_complexity_pipeline(R):
    """Verify the full pipeline convenience function."""
    print("\n--- Full Pipeline ---")

    sv = random_score_vectors(15, 4, 3, seed=999)
    emb = score_embedding(sv)
    # Calibrate lambda from an independent reference sample
    emb_ref = score_embedding(random_score_vectors(50, 4, 3, seed=0))
    lam = calibrate_bandwidth(emb_ref)
    result = compute_complexity(emb, lambda_=lam)

    # Check all keys present
    R.check(
        "pipeline_keys",
        set(result.keys()) == {'C', 'K', 'D', 'eigenvalues', 'lambda'},
        f"keys = {set(result.keys())}",
    )

    # Check shapes
    N = 15
    R.check("pipeline_C_scalar", np.isscalar(result['C']))
    R.check("pipeline_K_shape", result['K'].shape == (N, N))
    R.check("pipeline_D_shape", result['D'].shape == (N, N))
    R.check("pipeline_eigs_shape", result['eigenvalues'].shape == (N,))
    # Eigenvalues decreasing
    eigs = result['eigenvalues']
    R.check(
        "pipeline_eigs_decreasing",
        np.all(np.diff(eigs) <= 1e-10),
    )

    # C > 0
    R.check("pipeline_C_positive", result['C'] > 0, f"C={result['C']:.6f}")

    # Kernel swappable: Laplacian
    result_lap = compute_complexity(emb, lambda_=lam, kernel_fn=laplacian_kernel)
    R.check(
        "pipeline_laplacian_works",
        result_lap['C'] > 0,
        f"C_laplacian={result_lap['C']:.6f}",
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Experiment 1: Algebraic Property Verification")
    print("=" * 60)

    R = Results()

    test_permutation_invariance(R)
    test_bounded_monotonicity(R)
    test_monotonicity_extremes(R)
    test_sub_additivity(R)
    test_submodularity(R)
    test_identical_points(R)
    test_maximally_diverse(R)
    test_m_clusters(R)
    test_kernel_psd(R)
    test_cnd_d_squared(R)
    test_lipschitz_stability(R)
    test_compute_complexity_pipeline(R)

    all_passed = R.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
