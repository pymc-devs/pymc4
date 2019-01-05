from . import _template_contexts as contexts

import tensorflow as tf
import tensorflow_probability as tfp

__all__ = ["Normal", "HalfNormal", "Multinomial", "Dirichlet"]


class WithBackendArithmetic:
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
    _base_dist = None

    def __init__(self, name, *args, **kwargs):
        self._parents = []
        self._distribution = self._base_dist(*args, **kwargs)
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
            raise ValueError("Can not convert to tensor under new context.")
        if self._backend_tensor is None:
            self._backend_tensor = ctx.var_as_backend_tensor(self)
        return self._backend_tensor


class Normal(RandomVariable):
    _base_dist = tfp.distributions.Normal


class HalfNormal(RandomVariable):
    _base_dist = tfp.distributions.HalfNormal


class Multinomial(RandomVariable):
    _base_dist = tfp.distributions.Multinomial


class Dirichlet(RandomVariable):
    _base_dist = tfp.distributions.Dirichlet
