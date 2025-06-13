"""
Acquisition functions:  Expected Improvement (EI) only.
"""

import numpy as np
from .gaussian_process import GaussianProcess


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcess,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    mu, var = gp.predict(X, return_var=True)
    sigma = np.sqrt(var)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * _Phi(Z) + sigma * _phi(Z)
    ei[sigma < 1e-12] = 0.0
    return ei
