from . import _template_contexts as contexts

import tensorflow as tf
import tensorflow_probability as tfp

__all__ = ['Normal', 'HalfNormal']


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


class RandomVariable(WithBackendArithmetic):
    def __init__(self, name, *args, **kwargs):
        self._parents = []
        self._distribution = None
        self._sample_shape = ()
        self._dim_names = ()
        self.name = name
        ctx = contexts.get_context()
        self._creation_context_id = id(ctx)
        self._backend_tensor = None
        ctx.add_variable(self)

    def sample(self):
        return self._distribution.sample()

    def as_tensor(self):
        ctx = contexts.get_context()
        if id(ctx) != self._creation_context_id:
            raise ValueError('Can not convert to tensor '
                             'under new context.')
        if self._backend_tensor is None:
            self._backend_tensor = ctx.var_as_backend_tensor(self)
        return self._backend_tensor

    def __mul__(self, other):
        return self.as_tensor() * other

    def __div__(self, other):
        return self.as_tensor() / other

    def __add__(self, other):
        return self.as_tensor() + other


class Normal(RandomVariable):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._distribution = tfp.distributions.Normal(*args, **kwargs)


class HalfNormal(RandomVariable):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._distribution = tfp.distributions.HalfNormal(*args, **kwargs)

