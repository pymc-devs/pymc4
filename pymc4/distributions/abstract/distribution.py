import abc
import copy
from typing import Optional, Union
from . import transforms
from pymc4.coroutine_model import Model, unpack

NameType = Union[str, int]


class Distribution(Model):
    """Statistical distribution.

    An abstract class with consistent API across backends
    """

    def __init__(
        self, name: Optional[NameType], *, transform=None, observed=None, plate=None, **kwargs
    ):
        self.conditions = self.unpack_conditions(**kwargs)
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
        self._init_backend()

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

    @abc.abstractmethod
    def _init_backend(self):
        """Initialize the backend."""
        pass

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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

    @abc.abstractmethod
    def log_prob(self, value):
        """Return log probability in backend array format."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob_numpy(self, value):
        """Return log probability in numpy array format."""
        raise NotImplementedError

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
            return transforms.Log.create()
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


class Potential(object):
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
    @abc.abstractmethod
    def value_numpy(self):
        raise NotImplementedError


class Deterministic(Model):
    """An object that can be sampled, but has no log probability."""

    def __init__(self, name: Optional[NameType], value):
        super().__init__(
            self.unpack_distribution, name=name, keep_return=True, keep_auxiliary=False
        )
        self._init_backend()
        self.value = value

    def unpack_distribution(self):
        return unpack(self.value)

    @abc.abstractmethod
    def _init_backend(self):
        """Initialize the backend."""
        pass
