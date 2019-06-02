import abc
from ..coroutine_model import Model, unpack


class Distribution(Model):
    def __init__(self, name, keep_auxiliary=False, keep_return=True, transform=None, **kwargs):
        self.conditions = self.unpack_conditions(**kwargs)
        super().__init__(self.unpack_distribution, name=name, keep_return=keep_return, keep_auxiliary=keep_auxiliary)
        self.transform = transform

    def unpack_distribution(self):
        return unpack(self)

    @staticmethod
    def unpack_conditions(**kwargs) -> dict:
        """
        Parse arguments
        """
        return kwargs

    def sample(self, shape=(), seed=None):
        """
        Forward sampling implementation

        Parameters
        ----------
        shape : tuple
            sample shape
        seed : int|None
            random seed
        """
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError


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
        return 0.

    def upper_limit(self):
        return 1.


class PositiveContinuousDistribution(BoundedContinuousDistribution):
    def lower_limit(self):
        return 0.

    def upper_limit(self):
        return float("inf")


class PositiveDiscreteDistribution(BoundedDiscreteDistribution):
    def lower_limit(self):
        return 0

    def upper_limit(self):
        return float("inf")


class SimplexContinuousDistribution(ContinuousDistribution):
    ...
