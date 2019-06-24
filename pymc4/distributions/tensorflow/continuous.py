"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""
import tensorflow_probability as tfp
tfd = tfp.distributions


from .distribution import BackendDistribution
from ..abstract.continuous import Normal, HalfNormal

__all__ = ['Normal', 'HalfNormal']


class Normal(BackendDistribution, Normal):
    r"""
    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    """
    __doc__ = Normal.__doc__ + __doc__

    def _init_backend(self):
        mu, sigma = self.conditions['mu'], self.conditions['sigma']
        self._backend_distribution = tfd.Normal(
            loc=mu, scale=sigma, name=self.name)


class HalfNormal(BackendDistribution, HalfNormal):
    r"""
    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - sigma: scale
    """
    __doc__ = HalfNormal.__doc__ + __doc__

    def _init_backend(self):
        sigma = self.conditions['sigma']
        self._backend_distribution = tfd.HalfNormal(scale=sigma, name=self.name)
