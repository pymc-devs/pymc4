from typing import Union

import tensorflow as tf
import numpy as np


ArrayLike = Union[np.ndarray, tf.Tensor]
TfTensor = tf.Tensor
FreeRV = ArrayLike


def stabilize(K, shift=1e-6):
    r"""Add a diagonal shift to a covarience matrix"""
    return tf.linalg.set_diag(K, tf.linalg.diag_part(K) + shift)


def _inherit_docs(frommeth):
    """Inherits docs from `frommeth`."""

    def inherit(tometh):
        methdocs = frommeth.__doc__
        if methdocs is None:
            raise ValueError("No docs to inherit!")
        tometh.__doc__ = methdocs
        return tometh

    return inherit
