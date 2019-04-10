"""
PyMC4 base random variable class.

Implements the RandomVariable base class and the necessary BackendArithmetic.

Also stores the type hints used in child classes.
- TensorLike is for float-like tensors (scalars, vectors, matrices, tensors)
- IntTensorLike like TensorLike, just for ints.
"""

from .. import _template_contexts as contexts
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors
from typing import NewType, Union, Sequence


class WithBackendArithmetic:
    """Helper class to implement the backend arithmetic necessary for the RandomVariable class."""

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

    def __pow__(self, other):
        return self.as_tensor() ** other

    def __rpow__(self, other):
        return other ** self.as_tensor()

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
    """Random variable base class.

    Random variables must support 1) sampling, 2) computation of the log
    probability, and 3) conversion to tensors.

    The base distribution is transformed automatically by a default "identity"
    bijector transform. For other classes of distributions, we can pass in
    alternate transforms. The transformed distribution is used only when
    calculating the log_prob, while the base distribution is used for sampling
    purposes.
    """

    _bijector = bijectors.Identity()
    _base_dist = None

    def __init__(self, *args, **kwargs):
        self._parents = []
        self._untransformed_distribution = self._base_dist(*args, **kwargs)
        self._sample_shape = ()
        self._dim_names = ()
        ctx = contexts.get_context()
        self.name = kwargs.get("name", None)
        if isinstance(ctx, contexts.InferenceContext) and self.name is None:
            # Unfortunately autograph does not allow changing the AST,
            # thus we instead retrieve the name from when it was set
            # ForwardContext where AST parsing is possible.
            order_id = len(ctx.vars)  # where am I in the order of RV creation?
            self.name = ctx._names[order_id]

        if not isinstance(ctx, contexts.FreeForwardContext) and self.name is None:
            # We only require names for book keeping during inference
            raise ValueError("No name was set. Supply one via the name kwarg.")

        self._creation_context_id = id(ctx)
        self._backend_tensor = None
        # Override default bijector if provided
        self._bijector = kwargs.get("bijector", self._bijector)

        self._distribution = tfd.TransformedDistribution(
            distribution=self._untransformed_distribution, bijector=bijectors.Invert(self._bijector)
        )
        ctx.add_variable(self)

    def sample(self):
        """Forward sampling from the base distribution, unconditioned on data."""
        return self._untransformed_distribution.sample()

    def log_prob(self):
        """Log probability computation.

        Done based on the transformed distribution, not the base distribution.
        """
        return self._distribution.log_prob(self)

    def as_tensor(self):
        ctx = contexts.get_context()
        if id(ctx) != self._creation_context_id:
            raise ValueError("Cannot convert to tensor under new context.")
        if self._backend_tensor is None:
            self._backend_tensor = ctx.var_as_backend_tensor(self)

        return self._bijector.forward(self._backend_tensor)


class PositiveContinuousRV(RandomVariable):
    _bijector = bijectors.Exp()


class UnitContinuousRV(RandomVariable):
    _bijector = bijectors.Sigmoid()


TensorLike = NewType("TensorLike", Union[Sequence[int], Sequence[float], int, float])
IntTensorLike = NewType("IntTensorLike", Union[int, Sequence[int]])
