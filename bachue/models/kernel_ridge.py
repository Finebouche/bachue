"""
Kernel ridge regression on a precomputed kernel (bachue.models.volterra_kernel), with an unpenalized bias: the kernel
of the centered features, centered with the means of the training rows (as the dual GeneralizedRidge of
bachue.models.generalized_readout). The spectrum of the centered training kernel is kept: the coefficients of another
ridge coefficient cost one product (choice of the coefficient by cross-validation).

Example (kernels K of the training rows, K_test of the test rows with them):
    >>> readout = KernelRidge(K, Y)
    >>> scores = readout.predict(K_test, ridge=1e-3)
"""
import numpy as np


class KernelRidge:
    def __init__(self, K, Y):
        K, Y = np.asarray(K, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        self.y_mean = Y.mean(axis=0)
        self.column_means, self.mean = K.mean(axis=0), K.mean()
        centered = K + self.mean - self.column_means[:, None] - self.column_means[None]
        eigenvalues, self.eigenvectors = np.linalg.eigh(centered)
        self.eigenvalues = np.maximum(eigenvalues, 0)  # (rounding errors of a positive semidefinite matrix)
        self.projected = self.eigenvectors.T @ (Y - self.y_mean)

    def coefficients(self, ridge):
        """Dual coefficients (n, outputs) for this ridge coefficient (without ridge, the negligible eigenvalues
        ignored, as np.linalg.pinv)."""
        denominator = self.eigenvalues + ridge
        tolerance = self.eigenvalues.max() * np.finfo(float).eps * len(self.eigenvalues)
        inverse = np.divide(1.0, denominator, out=np.zeros_like(denominator), where=denominator > tolerance)
        return self.eigenvectors @ (self.projected * inverse[:, None])

    def centered(self, K):
        """The kernel K (m, n) of rows with the training rows, centered as the training kernel."""
        K = np.asarray(K, dtype=np.float64)
        return K + self.mean - K.mean(axis=1, keepdims=True) - self.column_means[None]

    def predict(self, K, ridge):
        """Outputs of the rows of the kernel K (m, n) with the training rows."""
        return self.centered(K) @ self.coefficients(ridge) + self.y_mean

    def predictions(self, K, ridges):
        """Yields (ridge, outputs of the rows of the kernel K) for each coefficient of ridges (the kernel projected on
        the eigenvectors once)."""
        projected = self.centered(K) @ self.eigenvectors
        for ridge in ridges:
            denominator = self.eigenvalues + ridge
            tolerance = self.eigenvalues.max() * np.finfo(float).eps * len(self.eigenvalues)
            inverse = np.divide(1.0, denominator, out=np.zeros_like(denominator), where=denominator > tolerance)
            yield ridge, (projected * inverse[None]) @ self.projected + self.y_mean
