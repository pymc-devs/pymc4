"""
PyMC4 discrete random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""
import tensorflow_probability as tfp
from pymc4.distributions import abstract
from pymc4.distributions.tensorflow.distribution import BackendDistribution


tfd = tfp.distributions

__all__ = ['Bernoulli',  'Binomial',  'Constant',
           'Poisson', 'NegativeBinomial', 'Geometric', 'Categorical']


class Bernoulli(BackendDistribution, abstract.Bernoulli):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - p: probs
    """.format(
        abstract.Bernoulli.__doc__
    )

    def _init_backend(self):
        p = self.conditions["p"]
        self._backend_distribution = tfd.Bernoulli(probs=p, name=self.name)


class Binomial(BackendDistribution, abstract.Binomial):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - n: total_count
    - p: probs
    """.format(
        abstract.Binomial.__doc__
    )

    def _init_backend(self):
        n, p = self.conditions["n"], self.conditions["p"]
        self._backend_distribution = tfd.Binomial(
            total_count=n, probs=p, name=self.name)


class Constant(BackendDistribution, abstract.Constant):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - value: loc
    """
    def _init_backend(self):
        value = self.conditions["value"]
        self._backend_distribution = tfd.Deterministic(
            loc=value, name=self.name)


class Poisson(BackendDistribution, abstract.Poisson):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: rate
    """.format(
        abstract.Poisson.__doc__
    )

    def _init_backend(self):
        mu = self.conditions["mu"]
        self._backend_distribution = tfd.Poisson(rate=mu, name=self.name)


class NegativeBinomial(BackendDistribution, abstract.NegativeBinomial):
    __doc__ = r"""{}


    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu + alpha: total_count
    - mu / (mu + alpha): probs
    """.format(
        abstract.NegativeBinomial.__doc__
    )

    def _init_backend(self):
        mu, alpha = self.conditions["mu"], self.conditions["alpha"]
        total_count = mu + alpha
        probs = mu / (mu + alpha)
        self._backend_distribution = tfd.NegativeBinomial(
            total_count=total_count, probs=probs, name=self.name)


class Geometric(BackendDistribution, abstract.Geometric):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - p: probs
    """.format(
        abstract.Geometric.__doc__
    )

    def _init_backend(self):
        p = self.conditions["p"]
        self._backend_distribution = tfd.Geometric(probs=p, name=self.name)


class Categorical(BackendDistribution, abstract.Categorical):
    __doc__ = r"""{}

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - p: probs
    """.format(
        abstract.Geometric.__doc__
    )

    def _init_backend(self):
        p = self.conditions["p"]
        self._backend_distribution = tfd.Categorical(probs=p, name=self.name)
