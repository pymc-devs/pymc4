from typing import Union

import tensorflow as tf
import numpy as np


ArrayLike = Union[np.ndarray, tf.python.framework.ops.EagerTensor]
TfTensor = tf.python.framework.ops.EagerTensor
FreeRV = ArrayLike


def stabilize(K, shift=1e-6):
    r"""Add a diagonal shift to a covarience matrix"""
    return tf.linalg.set_diag(K, tf.linalg.diag_part(K) + shift)
