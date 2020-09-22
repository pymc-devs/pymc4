import abc
import copy
import warnings
from typing import Optional, Union, Any, Tuple

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static

from pymc4.coroutine_model import Model, unpack
from pymc4.distributions.batchstack import BatchStacker
from pymc4.distributions import transforms

NameType = Union[str, int]

__all__ = (
    "Distribution",
    "Potential",
    "ContinuousDistribution",
    "DiscreteDistribution",
    "PositiveContinuousDistribution",
    "PositiveDiscreteDistribution",
    "BoundedDistribution",
    "BoundedDiscreteDistribution",
    "BoundedContinuousDistribution",
    "UnitContinuousDistribution",
    "SimplexContinuousDistribution",
)


class Distribution(Model):
    """Statistical distribution."""

    _grad_support: bool = True
    _test_value = 0.0
    _base_parameters = ["dtype", "validate_args", "allow_nan_stats"]

    def __init__(
        self,
        name: Optional[NameType],
        *,
        transform=None,
        observed=None,
        batch_stack=None,
        event_stack=None,
        conditionally_independent=False,
        reinterpreted_batch_ndims=0,
        dtype=None,
        validate_args=False,
        allow_nan_stats=False,
        **kwargs,
    ):
        self.conditions, self.base_parameters = self.unpack_conditions(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            **kwargs,
        )
        self._distribution = self._init_distribution(self.conditions, **self.base_parameters)
        self._default_new_state_part = None
        super().__init__(
            self.unpack_distribution, name=name, keep_return=True, keep_auxiliary=False
        )
        if name is None and observed is not None:
            raise ValueError(
                "Observed variables are not allowed for anonymous (with name=None) Distributions"
            )
        self.model_info.update(observed=observed)
        self.transform = self._init_transform(transform)
        self.batch_stack = batch_stack
        self.event_stack = event_stack
        self.conditionally_independent = conditionally_independent
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        if reinterpreted_batch_ndims:
            self._distribution = tfd.Independent(
                self._distribution, reinterpreted_batch_ndims=reinterpreted_batch_ndims
            )
        if batch_stack is not None:
            self._distribution = BatchStacker(self._distribution, batch_stack=batch_stack)
        if event_stack is not None:
            self._distribution = tfd.Sample(self._distribution, sample_shape=self.event_stack)

        if self.transform is not None and self.transform.event_ndims is None:
            event_ndims = prefer_static.rank_from_shape(
                self._distribution.event_shape_tensor, self._distribution.event_shape
            )
            self.transform.event_ndims = event_ndims

    @property
    def dtype(self):
        return self._distribution.dtype

    @staticmethod
    def _init_distribution(conditions: dict, **kwargs) -> tfd.Distribution:
        ...

    def _init_transform(self, transform):
        return transform

    def unpack_distribution(self):
        return unpack(self)

    @classmethod
    def unpack_conditions(cls, **kwargs) -> Tuple[dict, dict]:
        """
        Parse arguments.

        This is used to form :attr:`conditions` for a distributions, as
        one may desire to have different parametrizations, this all should be done there
        """
        base_parameters = {k: v for k, v in kwargs.items() if k in cls._base_parameters}
        if base_parameters["dtype"] is None:
            del base_parameters["dtype"]
        conditions = {k: v for k, v in kwargs.items() if k not in cls._base_parameters}
        return conditions, base_parameters

    @property
    def test_value(self):
        return tf.cast(
            tf.broadcast_to(self._test_value, self.batch_shape + self.event_shape),
            self.dtype,
        )

    def sample(self, sample_shape=(), seed=None):
        """
        Forward sampling implementation.

        Parameters
        ----------
        sample_shape : tuple
            sample shape
        seed : int|None
            random seed
        """
        return self._distribution.sample(sample_shape, seed)

    def sample_numpy(self, sample_shape=(), seed=None):
        """
        Forward sampling implementation returning raw numpy arrays.

        Parameters
        ----------
        sample_shape : tuple
            sample shape
        seed : int|None
            random seed
        Returns
        ----------
        array of given shape
        """
        return self.sample(sample_shape, seed).numpy()

    def get_test_sample(self, sample_shape=(), seed=None):
        """
        Get the test value using a function signature similar to meth:`~.sample`.

        Parameters
        ----------
        sample_shape : tuple
            sample shape
        seed : int | None
            ignored. Is only present to match the signature of meth:`~.sample`

        Returns
        -------
        The distribution's ``test_value`` broadcasted to
        ``sample_shape + self.batch_shape + self.event_shape``
        """
        sample_shape = tf.TensorShape(sample_shape)
        return tf.broadcast_to(self.test_value, sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        """Return log probability as tensor."""
        return self._distribution.log_prob(value)

    def log_prob_numpy(self, value):
        """Return log probability in numpy array format."""
        return self.log_prob(value).numpy()

    @classmethod
    def dist(cls, *args, **kwargs):
        """Create an anonymous Distribution that can't be used within yield statement."""
        return cls(None, *args, **kwargs)

    def prior(self, name: NameType, *, transform=None, observed=None):
        """Finalize instantiation of an anonymous Distribution.

        The resulting distribution will have the provided name, transform and an observed variable
        making it act as a prior and allow to participate within yield
        """
        if not self.is_anonymous:
            raise TypeError("Distribution is already not anonymous and cant define a new prior")
        if name is None:
            raise ValueError("Can't create a prior Distribution without a name")
        # internally is is ok to make a shallow copy of a distribution
        cloned_dist = copy.copy(self)
        # some mutable variables are required to be copied as well
        cloned_dist.model_info = cloned_dist.model_info.copy()
        cloned_dist.model_info.update(observed=observed)
        cloned_dist.conditions = cloned_dist.conditions.copy()
        if transform is not None:
            cloned_dist.transform = transform
        cloned_dist.name = cloned_dist.validate_name(name)
        return cloned_dist

    @property
    def is_anonymous(self):
        return self.name is None

    @property
    def is_observed(self):
        return self.model_info["observed"] is not None

    @property
    def is_root(self):
        return self.conditionally_independent

    @property
    def batch_shape(self):
        return self._distribution.batch_shape

    @property
    def event_shape(self):
        return self._distribution.event_shape

    @property
    def validate_args(self):
        return self._distribution.validate_args

    @property
    def allow_nan_stats(self):
        return self._distribution.allow_nan_stats


class Potential:
    __slots__ = ("_value", "_coef")

    def __init__(self, value, coef=1.0):
        self._value = value
        self._coef = coef

    @property
    def value(self):
        if callable(self._value):
            return self._value() * self._coef
        else:
            return self._value * self._coef

    @property
    def value_numpy(self):
        return self.value.numpy()


class Deterministic(Model):
    """An object that can be sampled, but has no log probability."""

    __slots__ = ("value",)

    def __init__(self, name: Optional[NameType], value: Any):
        self.value = value
        super().__init__(self.get_value, name=name, keep_return=True, keep_auxiliary=False)

    def get_value(self):
        return self.value

    @property
    def value_numpy(self):
        return self.value.numpy()

    @property
    def is_anonymous(self):
        return self.name is None


class ContinuousDistribution(Distribution):
    _test_value = 0.0

    @classmethod
    def unpack_conditions(cls, **kwargs):
        conditions, base_parameters = super().unpack_conditions(**kwargs)
        dtype = base_parameters.pop("dtype", None)
        if dtype is not None:
            warnings.warn(
                f"At the moment, the continuous distributions of the backend used by pymc4 "
                f"(tensorflow_probability) do not accept an explicit `dtype`. The supplied "
                f"dtype={dtype} will be ignored."
            )
        return conditions, base_parameters


class DiscreteDistribution(Distribution):
    _test_value = 0


class BoundedDistribution(Distribution):
    @abc.abstractmethod
    def lower_limit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def upper_limit(self):
        raise NotImplementedError


class BoundedDiscreteDistribution(DiscreteDistribution, BoundedDistribution):
    @property
    def _test_value(self):
        return tf.cast(tf.round(0.5 * (self.upper_limit() + self.lower_limit())), self.dtype)


class BoundedContinuousDistribution(ContinuousDistribution, BoundedDistribution):
    def _init_transform(self, transform):
        if transform is None:
            return transforms.Interval(self.lower_limit(), self.upper_limit())
        else:
            return transform

    @property
    def _test_value(self):
        return 0.5 * (self.upper_limit() + self.lower_limit())


class UnitContinuousDistribution(BoundedContinuousDistribution):
    def _init_transform(self, transform):
        if transform is None:
            return transforms.Sigmoid()
        else:
            return transform

    def lower_limit(self):
        return 0.0

    def upper_limit(self):
        return 1.0


class PositiveContinuousDistribution(BoundedContinuousDistribution):
    _test_value = 1.0

    def _init_transform(self, transform):
        if transform is None:
            return transforms.Log()
        else:
            return transform

    def lower_limit(self):
        return 0.0

    def upper_limit(self):
        return float("inf")


class PositiveDiscreteDistribution(BoundedDiscreteDistribution):
    _test_value = 1

    def lower_limit(self):
        return 0

    def upper_limit(self):
        return float("inf")


class SimplexContinuousDistribution(ContinuousDistribution):
    @property
    def test_value(self):
        return tf.ones(self.batch_shape + self.event_shape) / self.event_shape[-1]
