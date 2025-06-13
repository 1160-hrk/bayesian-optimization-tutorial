import numpy as np
from bayesian_optimization_tutorial import BayesOptimizer
from bayesian_optimization_tutorial.core.kernels import RBF
from bayesian_optimization_tutorial.datasets.branin import branin, BOUNDS

np.random.seed(0)

bo = BayesOptimizer(bounds=BOUNDS, kernel=RBF(length_scale=2.0))
x_best, y_best = bo.optimize(branin, n_iter=25)

print("\nFound minimum:", x_best, "→", y_best)
