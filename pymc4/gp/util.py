from typing import Union

import tensorflow as tf
import numpy as np


ArrayLike = Union[np.ndarray, tf.Tensor]
TfTensor = tf.Tensor
FreeRV = ArrayLike


def stabilize(K, shift=None):
    r"""Add a diagonal shift to a covariance matrix."""
    K = tf.convert_to_tensor(K)
    diag = tf.linalg.diag_part(K)
    if shift is None:
        shift = 1e-6 if K.dtype == tf.float64 else 1e-4
    return tf.linalg.set_diag(K, diag + shift)


def _inherit_docs(frommeth):
    r"""Decorate a method or class to inherit docs from `frommeth`."""

    def inherit(tometh):
        methdocs = frommeth.__doc__
        if methdocs is None:
            raise ValueError("No docs to inherit!")
        tometh.__doc__ = methdocs
        return tometh

    return inherit


def _build_docs(**doc_strings):
    r"""Decorate a method or class to build its doc strings."""

    def _builder(meth_or_cls):
        docs = meth_or_cls.__doc__
        meth_or_cls.__doc__ = docs % doc_strings
        return meth_or_cls

    return _builder
