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


def _build_docs(meth_or_cls):
    r"""Decorate a method or class to build its doc strings."""
    pattern = re.compile("\%\(.*\)")
    modname = inspect.getmodule(meth_or_cls)
    docs = meth_or_cls.__doc__
    while pattern.search(docs) is not None:
        docname = pattern.search(docs).group(0)[2:-1]
        try:
            docstr = getattr(modname, docname)
        except AttributeError:
            warnings.warn(
                f"While documenting {meth_or_cls.__name__}, arrtibute {docname} not found.",
                SyntaxWarning,
            )
            # FIXME: This should continue execution by skipping
            # the docs not found. Instead, currently, it just stops
            # execution!
            break
        docs = pattern.sub(docstr, docs, count=1)
    meth_or_cls.__doc__ = docs
    return meth_or_cls
