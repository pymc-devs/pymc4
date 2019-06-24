import abc
from ...coroutine_model import Model, unpack


class Distribution(Model):
    """Statistical distribution"""

    def __init__(self, name, keep_auxiliary=False, keep_return=True, transform=None, **kwargs):
        self.conditions = self.unpack_conditions(**kwargs)
        super().__init__(
            self.unpack_distribution,
            name=name,
            keep_return=keep_return,
            keep_auxiliary=keep_auxiliary,
        )
        self.transform = transform
        self._init_backend()

    def unpack_distribution(self):
        return unpack(self)

    @staticmethod
    def unpack_conditions(**kwargs) -> dict:
        """
        Parse arguments
        """
        return kwargs

    @abc.abstractmethod
    def _init_backend(self):
        """Initialize the backend."""
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def sample_numpy(self, shape=(), seed=None):
        """
        Forward sampling implementation returning raw numpy arrays

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
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob_numpy(self, value):
        """Return log probability in numpy array format"""
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
        return 0.0

    def upper_limit(self):
        return 1.0


class PositiveContinuousDistribution(BoundedContinuousDistribution):
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


class Potential(Distribution):
    def __init__(self, value):
        super().__init__(name=None)
        self.value = value

    def sample(self, shape=(), seed=None):
        raise NotImplementedError("Unavailable for Potential")

    def sample_numpy(self, shape=(), seed=None):
        raise NotImplementedError("Unavailable for Potential")

    def log_prob(self, value):
        raise NotImplementedError("Unavailable for Potential")

    def log_prob_numpy(self, value):
        raise NotImplementedError("Unavailable for Potential")

    @property
    @abc.abstractmethod
    def value_numpy(self):
        raise NotImplementedError
