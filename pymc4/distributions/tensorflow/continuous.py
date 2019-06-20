"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""
import tensorflow_probability as tfp
tfd = tfp.distributions


from .distribution import BackendDistribution
from ..base.continuous import Normal

__all__ = ['Normal']


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
