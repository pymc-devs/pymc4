"""
Implements the RandomVariable base class (and the necessary BackendArithmetic).
Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

from . import _template_contexts as contexts

import sys
import tensorflow_probability.distributions as tfd


# Must match tfp.distributions names exactly.
__all__ = [
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


# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in __all__:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
