"""
Bayesian-Optimization-Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure-NumPy, fully comment-driven code for learning Bayesian optimisation.
"""

from importlib.metadata import version as _v

__all__ = ["BayesOptimizer", "kernels", "acquisition"]
__version__ = _v("bayesian_optimization_tutorial")

from .core.loop import BayesOptimizer          # noqa: E402
from .core import kernels, acquisition         # noqa: E402
