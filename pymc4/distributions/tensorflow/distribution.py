import abc
from ...coroutine_model import Model, unpack
from ..backend import TensorflowBackend


class Distribution(Model, TensorflowBackend):

    def __init__(self,
                 name,
                 keep_auxiliary=False,
                 keep_return=True,
                 transform=None,
                 **kwargs):
        super().__init__(
            self.unpack_distribution,
            name=name,
            keep_auxiliary=keep_auxiliary,
            keep_return=keep_return,
        )
        self._init_backend()
        pass

    def unpack_distribution(self):
        return unpack(self)

    @abc.abstractmethod
    def _init_backend(self):
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
        Returns
        ---------
        array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, value):
        raise NotImplementedError


class ContinuousDistribution(Distribution):
    """Base class for continuous distributions"""

    def __init__(self,
                 name,
                 keep_auxiliary=False,
                 keep_return=True,
                 transform=None,
                 *args,
                 **kwargs):
        super().__init__(
            name=name,
            keep_auxiliary=keep_auxiliary,
            keep_return=keep_return,
            **kwargs)
