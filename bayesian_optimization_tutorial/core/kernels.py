"""
Basic covariance functions.
All kernels take (X, Y) with shapes (n, d) and (m, d) and return (n, m).
"""

import numpy as np
from typing import Protocol


class Kernel(Protocol):
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray: ...


class RBF:
    """Squared-exponential (a.k.a. Gaussian) kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.ell = float(length_scale)
        self.var = float(variance)

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        sqnorm = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(Y**2, axis=1)           # shape (m,)
            - 2.0 * X @ Y.T
        )
        return self.var * np.exp(-0.5 * sqnorm / self.ell**2)
