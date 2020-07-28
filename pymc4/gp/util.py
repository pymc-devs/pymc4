import inspect
import warnings
import re
from typing import Union

import tensorflow as tf
import numpy as np


ArrayLike = Union[np.ndarray, tf.Tensor]
TfTensor = tf.Tensor
FreeRV = ArrayLike


def stabilize(K, shift=None):
    r"""Add a diagonal shift to a covariance matrix."""
    diag = tf.linalg.diag_part(K)
    if shift is None:
        shifted = tf.math.nextafter(diag, np.inf)
        return tf.linalg.set_diag(K, shifted)
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


def _build_docs(**kwargs):
    r"""Decorate a method or class to build its doc strings."""

    def _doccer(meth_or_cls):
        if meth_or_cls.__doc__ is not None:
            meth_or_cls.__doc__ = meth_or_cls.__doc__ % kwargs
        return meth_or_cls

    return _doccer
