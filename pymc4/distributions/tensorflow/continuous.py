"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""
import tensorflow_probability as tfp
from pymc4.distributions import abstract
from pymc4.distributions.tensorflow.distribution import BackendDistribution


tfd = tfp.distributions

__all__ = ["Beta", "Cauchy", "HalfCauchy", "Normal", "HalfNormal"]

class Beta(BackendDistribution, abstract.Beta):
    __doc__ = r"""{}
    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: concentration1
    - beta: concentration0
    """.format(
        abstract.Beta.__doc__
    )
    __doc__ = abstract.Beta.__doc__ + __doc__

    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.Beta(
            concentration1=alpha, concentration0=beta, name=self.name)


class Cauchy(BackendDistribution, abstract.Cauchy):
    __doc__ = r"""{}
    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: loc
    - beta: scale
    """.format(
        abstract.Cauchy.__doc__
    )

    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.Cauchy(
            loc=alpha, scale=beta, name=self.name)


class HalfCauchy(BackendDistribution, abstract.HalfCauchy):
    __doc__ = r"""{}
    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: loc
    - beta: scale
    """.format(
        abstract.HalfCauchy.__doc__
    )

    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.HalfCauchy(
            loc=alpha, scale=beta, name=self.name)



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
        self._backend_distribution = tfd.Normal(loc=mu, scale=sigma)


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
        self._backend_distribution = tfd.HalfNormal(scale=sigma)
