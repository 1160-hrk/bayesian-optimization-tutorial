import numpy as np
from bayesian_optimization_tutorial.core.gaussian_process import GaussianProcess
from bayesian_optimization_tutorial.core.kernels import RBF

def test_gp_one_point_exact():
    X = np.array([[0.0]])
    y = np.array([1.23])
    gp = GaussianProcess(RBF()).fit(X, y)

    mu, var = gp.predict(X, return_var=True)
    assert np.allclose(mu, y)
    assert var < 1e-9
