"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""
import tensorflow_probability as tfp
from pymc4.distributions import abstract
from pymc4.distributions.tensorflow.distribution import BackendDistribution


tfd = tfp.distributions

__all__ = ["Normal", "HalfNormal"]


class Normal(BackendDistribution, abstract.Normal):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    """.format(
        abstract.Normal.__doc__
    )

    def _init_backend(self):
        mu, sigma = self.conditions["mu"], self.conditions["sigma"]
        self._backend_distribution = tfd.Normal(loc=mu, scale=sigma, name=self.name)


class HalfNormal(BackendDistribution, abstract.HalfNormal):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - sigma: scale
    """.format(
        abstract.HalfNormal.__doc__
    )

    def _init_backend(self):
        sigma = self.conditions["sigma"]
        self._backend_distribution = tfd.HalfNormal(scale=sigma, name=self.name)
