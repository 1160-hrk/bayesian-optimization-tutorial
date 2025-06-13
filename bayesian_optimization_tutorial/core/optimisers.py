import numpy as np
from typing import Sequence, Tuple


def sample_uniform(bounds: Sequence[Tuple[float, float]], n: int) -> np.ndarray:
    """Return (n, d) array uniformly sampled inside bounds."""
    bounds = np.asarray(bounds, float)
    low, high = bounds[:, 0], bounds[:, 1]
    return np.random.uniform(low, high, size=(n, len(bounds)))
