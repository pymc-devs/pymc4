"""
Implements the RandomVariable base class (and the necessary BackendArithmetic).
Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# pylint: disable=undefined-all-variable
from . import _template_contexts as contexts

import sys
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


# Random variables tfp does not support as distributions. We implement these
# random variables.
tfp_unsupported = [
    "Constant",
    "DiscreteUniform",
    "HalfStudentT",
    "LogitNormal",
    "Weibull",
    "ZeroInflatedBinomial",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]

# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "Dirichlet",
    "Exponential",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "InverseGaussian",
    "Kumaraswamy",
    "LKJ",
    "Laplace",
    "LogNormal",
    "Logistic",
    "Multinomial",
    "MultivariateNormalFullCovariance",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "StudentT",
    "Triangular",
    "Uniform",
    "VonMises",
    "Wishart",
]

__all__ = tfp_supported + tfp_unsupported


class WithBackendArithmetic:
    """
    Helper class to implement the backend arithmetic necessary for the RandomVariable class.
    """

    def __add__(self, other):
        return self.as_tensor() + other

    def __radd__(self, other):
        return other + self.as_tensor()

    def __sub__(self, other):
        return self.as_tensor() - other

    def __rsub__(self, other):
        return other - self.as_tensor()

    def __mul__(self, other):
        return self.as_tensor() * other

    def __rmul__(self, other):
        return other * self.as_tensor()

    def __matmul__(self, other):
        return self.as_tensor() @ other

    def __rmatmul__(self, other):
        return other @ self.as_tensor()

    def __truediv__(self, other):
        return self.as_tensor() / other

    def __rtruediv__(self, other):
        return other / self.as_tensor()

    def __floordiv__(self, other):
        return self.as_tensor() // other

    def __rfloordiv__(self, other):
        return other // self.as_tensor()

    def __mod__(self, other):
        return self.as_tensor() % other

    def __rmod__(self, other):
        return other % self.as_tensor()

    def __divmod__(self, other):
        return divmod(self.as_tensor(), other)

    def __rdivmod__(self, other):
        return divmod(other, self.as_tensor())

    def __pow__(self, other):
        return self.as_tensor() ** other

    def __rpow__(self, other):
        return other ** self.as_tensor()

    def __lshift__(self, other):
        return self.as_tensor() << other

    def __rlshift__(self, other):
        return other << self.as_tensor()

    def __rshift__(self, other):
        return self.as_tensor >> other

    def __rrshift__(self, other):
        return other >> self.as_tensor()

    def __and__(self, other):
        return self.as_tensor() & other

    def __rand__(self, other):
        return other & self.as_tensor()

    def __xor__(self, other):
        return self.as_tensor() ^ other

    def __rxor__(self, other):
        return other ^ self.as_tensor()

    def __or__(self, other):
        return self.as_tensor() | other

    def __ror__(self, other):
        return other | self.as_tensor()

    def __neg__(self):
        return -self.as_tensor()

    def __pos__(self):
        return +self.as_tensor()

    def __invert__(self):
        return ~self.as_tensor()

    def __getitem__(self, slice_spec, var=None):
        return self.as_tensor().__getitem__(slice_spec, var=var)


class RandomVariable(WithBackendArithmetic):
    """
    Random variable base class.

    Random variables must support 1) sampling, 2) computation of the log
    probability, and 3) conversion to tensors.
    """

    _base_dist = None

    def __init__(self, name, *args, **kwargs):
        self._parents = []
        self._distribution = self._base_dist(name=name, *args, **kwargs)
        self._sample_shape = ()
        self._dim_names = ()
        self.name = name
        ctx = contexts.get_context()
        self._creation_context_id = id(ctx)
        self._backend_tensor = None
        ctx.add_variable(self)

    def sample(self):
        return self._distribution.sample()

    def log_prob(self):
        return self._distribution.log_prob(self)

    def as_tensor(self):
        ctx = contexts.get_context()
        if id(ctx) != self._creation_context_id:
            raise ValueError("Cannot convert to tensor under new context.")
        if self._backend_tensor is None:
            self._backend_tensor = ctx.var_as_backend_tensor(self)

        return self._backend_tensor


# FIXME all RandomVariable classes need docstrings


class Constant(RandomVariable):
    _base_dist = tfd.Deterministic


class DiscreteUniform(RandomVariable):
    def __init__(self, name, low, high, *args, **kwargs):
        """Add `low` and `high` to kwargs."""
        kwargs.update({"low": low, "high": high})
        super(DiscreteUniform, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """A DiscreteUniform is an equiprobable Categorical over (high-low),
        shifted up by low."""
        low = kwargs.pop("low")
        high = kwargs.pop("high")
        probs = np.ones(high - low) / (high - low)
        return tfd.TransformedDistribution(
            distribution=tfd.Categorical(probs=probs),
            bijector=tfp.bijectors.AffineScalar(shift=low),
            name="DiscreteUniform",
        )


class HalfStudentT(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """A HalfStudentT is the absolute value of a StudentT."""
        return tfd.TransformedDistribution(
            distribution=tfd.StudentT(*args, **kwargs),
            bijector=tfp.bijectors.AbsoluteValue(),
            name="HalfStudentT",
        )


class LogitNormal(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """A LogitNormal is the standard logistic of a Normal."""
        return tfd.TransformedDistribution(
            distribution=tfd.Normal(*args, **kwargs),
            bijector=tfp.bijectors.Sigmoid(),
            name="LogitNormal",
        )


class Weibull(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """The inverse of the Weibull bijector applied to a U[0, 1] random
        variable gives a Weibull-distributed random variable."""
        return tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=0.0, high=1.0),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Weibull(*args, **kwargs)),
            name="Weibull",
        )


class ZeroInflatedBinomial(RandomVariable):
    def __init__(self, name, mix, *args, **kwargs):
        """Add `mix` to kwargs."""
        kwargs.update({"mix": mix})
        super(ZeroInflatedBinomial, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """A ZeroInflatedBinomial is a mixture between a deterministic
        distribution and a Binomial distribution."""
        mix = kwargs.pop("mix")
        return tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
            components=[tfd.Deterministic(0.0), tfd.Binomial(*args, **kwargs)],
            name="ZeroInflatedBinomial",
        )


class ZeroInflatedPoisson(RandomVariable):
    def __init__(self, name, mix, *args, **kwargs):
        """Add `mix` to kwargs."""
        kwargs.update({"mix": mix})
        super(ZeroInflatedPoisson, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """A ZeroInflatedPoisson is a mixture between a deterministic
        distribution and a Poisson distribution."""
        mix = kwargs.pop("mix")
        return tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
            components=[tfd.Deterministic(0.0), tfd.Poisson(*args, **kwargs)],
            name="ZeroInflatedPoisson",
        )


class ZeroInflatedNegativeBinomial(RandomVariable):
    def __init__(self, name, mix, *args, **kwargs):
        """Add `mix` to kwargs."""
        kwargs.update({"mix": mix})
        super(ZeroInflatedNegativeBinomial, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """A ZeroInflatedNegativeBinomial is a mixture between a deterministic
        distribution and a NegativeBinomial distribution."""
        mix = kwargs.pop("mix")
        return tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
            components=[tfd.Deterministic(0.0), tfd.NegativeBinomial(*args, **kwargs)],
            name="ZeroInflatedNegativeBinomial",
        )


# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
