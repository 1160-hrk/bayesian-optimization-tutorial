"""
Minimal Gaussian Process regression with analytic derivatives.

Only supports zero mean and homoskedastic noise σ_n^2 I.
"""

from __future__ import annotations
import numpy as np
from .kernels import Kernel


class GaussianProcess:
    def __init__(self, kernel: Kernel, noise: float = 1e-6):
        self.kernel = kernel
        self.noise = float(noise)

        # Will be set after .fit(...)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.L: np.ndarray | None = None    # Cholesky factor
        self.alpha: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Store data and pre-compute (K + σ_n²I)⁻¹ y via Cholesky."""
        X = np.ascontiguousarray(X, dtype=float)
        y = np.ascontiguousarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be (n, d) array")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        K = self.kernel(X, X) + self.noise**2 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

        self.X, self.y, self.L, self.alpha = X, y, L, alpha
        return self

    # ──────────────────────────────────────────────────────────
    def predict(
        self, X_star: np.ndarray, return_var: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return GP mean (and variance) at query points X_star."""
        if self.X is None:
            raise RuntimeError("GP not yet fitted")

        X_star = np.ascontiguousarray(X_star, dtype=float)
        K_s = self.kernel(self.X, X_star)
        mu = K_s.T @ self.alpha

        if not return_var:
            return mu

        # v = L⁻¹ K_s
        v = np.linalg.solve(self.L, K_s)
        K_ss = self.kernel(X_star, X_star)
        var = np.maximum(  # numerical safety
            np.diag(K_ss) - np.sum(v * v, axis=0),
            1e-12,
        )
        return mu, var
