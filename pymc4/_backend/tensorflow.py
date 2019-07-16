from pymc4._backend.base import Backend
import tensorflow as tf
import numpy as np
from typing import TypeVar, Tuple, Union, Any, Type

T = TypeVar("T")


class TensorflowBackend(Backend):
    @staticmethod
    def sum(a: T, axis: Union[int, Tuple[int]] = None, keepdims=False) -> T:
        return tf.reduce_sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def numpy(a: T) -> np.ndarray:
        return a.numpy()

    @staticmethod
    def tensor(a: Any, dtype: Type[np.dtype] = None) -> T:
        return tf.convert_to_tensor(a, dtype=dtype)
