import abc
import numpy as np
from typing import TypeVar, Tuple, Union, Any, Type


T = TypeVar("T")  # like a tensor dummy


class Backend(abc.ABC):
    """
    Abstract backend that defines common shared API for internal pymc4 needs.

    The backend should not be used by directly by users as this creates too much teaching overhead.
    Instead, this backend should be used by only the PyMC4 internals. The abstract base class ensures all
    necessary functions are implemented and share same API. Please follow numpy API style for new functions if
    you need them.
    """

    @staticmethod
    @abc.abstractmethod
    def sum(a: T, axis: Union[int, Tuple[int]] = None, keepdims=False) -> T:
        ...

    @staticmethod
    @abc.abstractmethod
    def numpy(a: T) -> np.ndarray:
        ...

    @staticmethod
    @abc.abstractmethod
    def tensor(a: Any, dtype: Type[np.dtype] = None) -> T:
        ...
