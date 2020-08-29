"""Mean functions for PyMC4's Gaussian Process Module."""

from typing import Union

import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util

from .util import ArrayLike, TfTensor, _inherit_docs

__all__ = ["Mean", "Zero", "Constant"]


class Mean:
    r"""
    Base Class for all the mean functions in GP.

    Parameters
    ----------
    feature_ndims : int, optional
        The number of feature dimensions to be absorbed during
        the computation. (default=1)
    """

    def __init__(self, feature_ndims=1):
        self.feature_ndims = feature_ndims

    def __call__(self, X: ArrayLike) -> TfTensor:
        r"""
        Evaluate the mean function at a point.

        Parameters
        ----------
        X : array_like
            Tensor or array of points at which to evaluate
            the mean function.

        Returns
        -------
        mu : tensorflow.Tensor
            Mean evaluated at points ``X``.
        """
        raise NotImplementedError("Your mean function should override this method.")

    def __add__(self, mean2):
        return MeanAdd(self, mean2)

    def __mul__(self, mean2):
        return MeanProd(self, mean2)


class MeanAdd(Mean):
    r"""
    Addition of two or more mean functions.

    Parameters
    ----------
    mean1 : Mean
        First mean function
    mean2 : Mean
        Second mean function
    """

    def __init__(self, mean1: Mean, mean2: Mean):
        if mean1.feature_ndims != mean2.feature_ndims:
            raise ValueError("Cannot combine means with different feature_ndims.")
        self.mean1 = mean1
        self.mean2 = mean2

    @_inherit_docs(Mean.__call__)
    def __call__(self, X: ArrayLike) -> TfTensor:
        return self.mean1(X) + self.mean2(X)


class MeanProd(Mean):
    r"""
    Product of two or more mean functions.

    Parameters
    ----------
    mean1 : Mean
        First mean function
    mean2 : Mean
        Second mean function
    """

    def __init__(self, mean1: Mean, mean2: Mean):
        if mean1.feature_ndims != mean2.feature_ndims:
            raise ValueError("Cannot combine means with different feature_ndims.")
        self.mean1 = mean1
        self.mean2 = mean2

    @_inherit_docs(Mean.__call__)
    def __call__(self, X: ArrayLike) -> TfTensor:
        return self.mean1(X) * self.mean2(X)


class Zero(Mean):
    r"""
    Zero mean function.

    Parameters
    ----------
    feature_ndims : int, optional
        number of rightmost dims to include in mean computation. (default=1)
    """

    @_inherit_docs(Mean.__call__)
    def __call__(self, X: ArrayLike) -> TfTensor:
        dtype = dtype_util.common_dtype([X])
        X = tf.convert_to_tensor(X, dtype=dtype)
        return tf.zeros(X.shape[: -self.feature_ndims], dtype=dtype)


class Constant(Mean):
    r"""
    Constant mean function.

    Parameters
    ----------
    coef : array_like, optional
        co-efficient to scale the mean. (default=1)
    feature_ndims : int, optional
        number of rightmost dims to include in mean computation. (default=1)
    """

    def __init__(self, coef: Union[ArrayLike, float] = 1, feature_ndims: int = 1):
        self.coef = coef
        super().__init__(feature_ndims=feature_ndims)

    @_inherit_docs(Mean.__call__)
    def __call__(self, X: ArrayLike) -> TfTensor:
        dtype = dtype_util.common_dtype([X, self.coef])
        X = tf.convert_to_tensor(X, dtype=dtype)
        return tf.ones(X.shape[: -self.feature_ndims], dtype=dtype) * self.coef
