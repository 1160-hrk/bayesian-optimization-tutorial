import numpy as np

BOUNDS = [(-5.0, 10.0), (0.0, 15.0)]

def branin(x: np.ndarray) -> float:
    x1, x2 = x
    a, b, c = 1.0, 5.1 / (4.0 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + \
           s * (1 - t) * np.cos(x1) + s
