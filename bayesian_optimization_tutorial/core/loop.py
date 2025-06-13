import numpy as np
from collections.abc import Callable
from .gaussian_process import GaussianProcess
from .acquisition import expected_improvement
from .optimisers import sample_uniform
from typing import Sequence, Tuple, List


class BayesOptimizer:
    """
    Ask–tell style Bayesian optimiser.
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        kernel,
        init_points: int = 5,
        acq_samples: int = 1000,
        random_state: int | None = None,
    ):
        self.rng = np.random.default_rng(random_state)
        self.bounds = np.asarray(bounds, float)
        self.gp = GaussianProcess(kernel)
        self.init_points = init_points
        self.acq_samples = acq_samples
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    # ──────────────────────────────────
    def ask(self) -> np.ndarray:
        if len(self.X) < self.init_points:
            return sample_uniform(self.bounds, 1)[0]

        # candidate pool
        X_cand = sample_uniform(self.bounds, self.acq_samples)
        y_best = min(self.y)
        ei = expected_improvement(X_cand, self.gp, y_best)
        return X_cand[np.argmax(ei)]

    # ──────────────────────────────────
    def tell(self, x_new: np.ndarray, y_new: float):
        self.X.append(np.asarray(x_new, float))
        self.y.append(float(y_new))
        self.gp.fit(np.vstack(self.X), np.array(self.y))

    # ──────────────────────────────────
    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        n_iter: int,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        for i in range(n_iter):
            x = self.ask()
            y = func(x)
            self.tell(x, y)
            if verbose:
                print(f"[{i:02d}] f({x}) = {y:.4g}   best={min(self.y):.4g}")
        best_idx = int(np.argmin(self.y))
        return self.X[best_idx], self.y[best_idx]
