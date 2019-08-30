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


class Beta(BackendDistribution, abstract.Beta):
    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.Beta(concentration0=alpha, concentration1=beta)


class Cauchy(BackendDistribution, abstract.Cauchy):
    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.Cauchy(loc=alpha, scale=beta)


class ChiSquared(BackendDistribution, abstract.ChiSquared):
    def _init_backend(self):
        nu = self.conditions["nu"]
        self._backend_distribution = tfd.Chi2(df=nu)


class Exponential(BackendDistribution, abstract.Exponential):
    def _init_backend(self):
        lam = self.conditions["lam"]
        self._backend_distribution = tfd.Exponential(rate=lam)


class Gamma(BackendDistribution, abstract.Gamma):
    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.Gamma(concentration=alpha, rate=beta)


class Gumbel(BackendDistribution, abstract.Gumbel):
    def _init_backend(self):
        mu, beta = self.conditions["mu"], self.conditions["beta"]
        self._backend_distribution = tfd.Gumbel(loc=mu, scale=beta)


class HalfCauchy(BackendDistribution, abstract.HalfCauchy):
    def _init_backend(self):
        beta = self.conditions["beta"]
        self._backend_distribution = tfd.HalfCauchy(loc=0, scale=beta)


class InverseGamma(BackendDistribution, abstract.InverseGamma):
    def _init_backend(self):
        alpha, beta = self.conditions["alpha"], self.conditions["beta"]
        self._backend_distribution = tfd.InverseGamma(concentration=alpha, scale=beta)


class InverseGaussian(BackendDistribution, abstract.InverseGaussian):
    def _init_backend(self):
        mu, lam = self.conditions["mu"], self.conditions["lam"]
        self._backend_distribution = tfd.InverseGaussian(loc=mu, concentration=lam)


class Kumaraswamy(BackendDistribution, abstract.Kumaraswamy):
    def _init_backend(self):
        a, b = self.conditions["a"], self.conditions["b"]
        self._backend_distribution = tfd.Kumaraswamy(concentration0=a, concentration1=b)


class Laplace(BackendDistribution, abstract.Laplace):
    def _init_backend(self):
        mu, b = self.conditions["mu"], self.conditions["b"]
        self._backend_distribution = tfd.Laplace(loc=mu, scale=b)


class Logistic(BackendDistribution, abstract.Logistic):
    def _init_backend(self):
        mu, s = self.conditions["mu"], self.conditions["s"]
        self._backend_distribution = tfd.Logistic(loc=mu, scale=s)


class LogitNormal(BackendDistribution, abstract.LogitNormal):
    def _init_backend(self):
        mu, sigma = self.conditions["mu"], self.conditions["sigma"]
        self._backend_distribution = tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=mu, scale=sigma),
            bijector=tfp.bijectors.Sigmoid(),
            name="LogitNormal",
        )


class LogNormal(BackendDistribution, abstract.LogNormal):
    def _init_backend(self):
        mu, sigma = self.conditions["mu"], self.conditions["sigma"]
        self._backend_distribution = tfd.LogNormal(loc=mu, scale=sigma)


class Pareto(BackendDistribution, abstract.Pareto):
    def _init_backend(self):
        alpha, m = self.conditions["alpha"], self.conditions["m"]
        self._backend_distribution = tfd.Pareto(concentration=alpha, scale=m)


class StudentT(BackendDistribution, abstract.StudentT):
    def _init_backend(self):
        nu, mu, sigma = self.conditions["nu"], self.conditions["mu"], self.conditions["sigma"]
        self._backend_distribution = tfd.StudentT(df=nu, loc=mu, scale=sigma)


class Triangular(BackendDistribution, abstract.Triangular):
    def _init_backend(self):
        lower, upper, c = self.conditions["lower"], self.conditions["upper"], self.conditions["c"]
        self._backend_distribution = tfd.Triangular(low=lower, high=upper, peak=c)


class Uniform(BackendDistribution, abstract.Uniform):
    def _init_backend(self):
        lower, upper = self.conditions["lower"], self.conditions["upper"]
        self._backend_distribution = tfd.Uniform(low=lower, high=upper)


class VonMises(BackendDistribution, abstract.VonMises):
    def _init_backend(self):
        mu, kappa = self.conditions["mu"], self.conditions["kappa"]
        self._backend_distribution = tfd.VonMises(loc=mu, concentration=kappa)
