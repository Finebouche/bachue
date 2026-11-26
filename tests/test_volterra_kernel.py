import itertools as it

import numpy as np
import pytest

from bachue.models.readout_rules.generalized_readout import GeneralizedRidge
from bachue.models.readout_rules.kernel_ridge import KernelRidge
from bachue.models.volterra_kernel import (clipped, lagged_instances, lagged_points, max_norm, parameters,
                                           volterra_gram)


def histories(rng, n, window, d):
    return lagged_instances([rng.uniform(-1, 1, size=(int(rng.integers(1, window + 3)), d)) for _ in range(n)], window)


def test_lagged_points_of_a_sequence():
    x = np.arange(10.0).reshape(5, 2)
    H = lagged_points(x, [-1, 0, 3], 3)
    np.testing.assert_array_equal(H[0], 0)
    np.testing.assert_array_equal(H[1], [x[0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(H[2], [x[3], x[2], x[1]])
    np.testing.assert_array_equal(lagged_instances([x[:4]], 3)[0], H[2])


def test_matches_the_series_14():
    rng = np.random.default_rng(0)
    A, B = histories(rng, 6, 5, 3), histories(rng, 4, 5, 3)
    tau, lam = parameters(0.7, 0.9, max(max_norm(A), max_norm(B)))
    G = np.einsum("iud,jud->iju", A, B)
    factors = 1 / (1 - tau ** 2 * G)  # (lag u; 1 beyond the window: zero frames)
    series, product = np.ones((len(A), len(B))), np.ones((len(A), len(B)))
    for k in range(1, 3000):
        product = product * (factors[:, :, k - 1] if k <= A.shape[1] else 1)
        series += lam ** (2 * k) * product
    np.testing.assert_allclose(volterra_gram(A, B, tau, lam), series, rtol=1e-12)


def test_matches_the_recursion_15_of_the_paper():
    rng = np.random.default_rng(1)
    z = rng.uniform(-1, 1, size=(12, 2))
    tau, lam = parameters(0.6, 0.8, max_norm(z[None]))
    n = len(z)
    K = np.full((n + 1, n + 1), 1 / (1 - lam ** 2))  # (row and column 0: before the first frame)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            K[i, j] = 1 + lam ** 2 * K[i - 1, j - 1] / (1 - tau ** 2 * z[i - 1] @ z[j - 1])
    H = lagged_points(z, np.arange(n), n)
    np.testing.assert_allclose(volterra_gram(H, H, tau, lam), K[1:, 1:], rtol=1e-12)


def test_is_the_inner_product_of_the_weighted_monomials():
    # (one input dimension, so that the tensor powers are the powers: the features of a frame z are tau^i z^i, those of
    # a history the products of those of its frames up to each delay, weighted by lambda^(delay + 1))
    rng = np.random.default_rng(2)
    window, degree = 3, 30
    A, B = histories(rng, 5, window, 1), histories(rng, 4, window, 1)
    tau, lam = parameters(0.5, 0.7, max(max_norm(A), max_norm(B)))

    def features(h):
        frames = [tau ** np.arange(degree + 1) * h[u, 0] ** np.arange(degree + 1) for u in range(window)]
        products = [frames[0]]
        for u in range(1, window):
            products.append(np.kron(products[-1], frames[u]))
        return [lam ** (j + 1) * p for j, p in enumerate(products)]

    def inner(a, b):
        fa, fb = features(a), features(b)
        value = 1 + sum(x @ y for x, y in zip(fa[:-1], fb[:-1]))
        # (the delays from window - 1 on: the same products, the older frames being zero, weighted lambda^(j + 1))
        return value + fa[-1] @ fb[-1] / (1 - lam ** 2)

    expected = np.array([[inner(a, b) for b in B] for a in A])
    np.testing.assert_allclose(volterra_gram(A, B, tau, lam), expected, rtol=1e-12)


def test_clipped_frames_keep_the_conditions():
    rng = np.random.default_rng(3)
    A = histories(rng, 5, 4, 3)
    norm = max_norm(A)
    B = clipped(3 * histories(rng, 5, 4, 3), norm)
    assert max_norm(B) <= norm * (1 + 1e-12)
    np.testing.assert_array_equal(clipped(A, norm), A)
    tau, lam = parameters(0.99, 0.99, norm)
    K = volterra_gram(np.concatenate([A, B]), np.concatenate([A, B]), tau, lam)
    assert np.isfinite(K).all() and np.linalg.eigvalsh(K).min() > -1e-8 * np.abs(K).max()


@pytest.mark.parametrize("ridge", [1e-6, 1e-2, 10.0])
def test_kernel_ridge_matches_the_primal_ridge(ridge):
    rng = np.random.default_rng(4)
    X, X_test = rng.normal(size=(40, 5)), rng.normal(size=(10, 5))
    Y = np.eye(3)[rng.integers(0, 3, 40)]
    primal = GeneralizedRidge(order=1, ridge=ridge, solver="primal").fit(X, Y)
    dual = KernelRidge(X @ X.T, Y)
    # (the dual kernel has rank 5 for 40 rows: with ridge 1e-6, rounding errors of order eps * largest eigenvalue /
    # ridge, about 1e-8)
    np.testing.assert_allclose(dual.predict(X_test @ X.T, ridge), primal.run(X_test), rtol=1e-6, atol=1e-7)


def test_predictions_of_several_ridges():
    rng = np.random.default_rng(5)
    X, X_test = rng.normal(size=(30, 40)), rng.normal(size=(8, 40))
    readout = KernelRidge(X @ X.T, np.eye(3)[rng.integers(0, 3, 30)])
    for ridge, outputs in readout.predictions(X_test @ X.T, [1e-3, 1.0, 100.0]):
        np.testing.assert_allclose(outputs, readout.predict(X_test @ X.T, ridge), rtol=1e-10, atol=1e-12)


def test_the_kernel_of_rows_with_themselves_is_symmetric(monkeypatch):
    import bachue.models.volterra_kernel as volterra
    monkeypatch.setattr(volterra, "BLOCK", 3)  # (several blocks)
    rng = np.random.default_rng(6)
    A = histories(rng, 10, 4, 2)
    tau, lam = parameters(0.6, 0.9, max_norm(A))
    np.testing.assert_allclose(volterra_gram(A, A, tau, lam), volterra_gram(A, A.copy(), tau, lam), rtol=1e-13)
