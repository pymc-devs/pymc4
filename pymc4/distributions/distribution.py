import abc
import copy
from typing import Optional, Union, Any

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from pymc4.coroutine_model import Model, unpack
from . import transforms

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

    def __init__(
        self, name: Optional[NameType], *, transform=None, observed=None, plate=None, **kwargs
    ):
        self.conditions = self.unpack_conditions(**kwargs)
        self._distribution = self._init_distribution(self.conditions)
        self.plate = plate
        super().__init__(
            self.unpack_distribution, name=name, keep_return=True, keep_auxiliary=False
        )
        if name is None and observed is not None:
            raise ValueError(
                "Observed variables are not allowed for anonymous (with name=None) Distributions"
            )
        self.model_info.update(observed=observed)
        self.transform = self._init_transform(transform)
        if self.plate is not None:
            self._distribution = tfd.Sample(self._distribution, sample_shape=self.plate)

    @staticmethod
    def _init_distribution(conditions: dict) -> tfd.Distribution:
        ...

    def _init_transform(self, transform):
        return transform

    def unpack_distribution(self):
        return unpack(self)

    @staticmethod
    def unpack_conditions(**kwargs) -> dict:
        """
        Parse arguments.

        This is used to form :attr:`conditions` for a distributions, as
        one may desire to have different parametrizations, this all should be done there
        """
        return kwargs

    def sample(self, shape=(), seed=None):
        """
        Forward sampling implementation.

        Parameters
        ----------
        shape : tuple
            sample shape
        seed : int|None
            random seed
        """
        return self._distribution.sample(shape, seed)

    def sample_numpy(self, shape=(), seed=None):
        """
        Forward sampling implementation returning raw numpy arrays.

        Parameters
        ----------
        shape : tuple
            sample shape
        seed : int|None
            random seed
        Returns
        ----------
        array of given shape
        """
        return self.sample(shape, seed).numpy()

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
        self.value = tf.identity(value)
        super().__init__(
            self.get_value, name=name, keep_return=True, keep_auxiliary=False
        )

    def get_value(self):
        return self.value

    @property
    def value_numpy(self):
        return self.value.numpy()

    @property
    def is_anonymous(self):
        return self.name is None


class ContinuousDistribution(Distribution):
    ...


class DiscreteDistribution(Distribution):
    ...


class BoundedDistribution(Distribution):
    @abc.abstractmethod
    def lower_limit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def upper_limit(self):
        raise NotImplementedError


class BoundedDiscreteDistribution(DiscreteDistribution, BoundedDistribution):
    ...


class BoundedContinuousDistribution(ContinuousDistribution, BoundedDistribution):
    ...


class UnitContinuousDistribution(BoundedContinuousDistribution):
    def lower_limit(self):
        return 0.0

    def upper_limit(self):
        return 1.0


class PositiveContinuousDistribution(BoundedContinuousDistribution):
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
    def lower_limit(self):
        return 0

    def upper_limit(self):
        return float("inf")


class SimplexContinuousDistribution(ContinuousDistribution):
    ...
