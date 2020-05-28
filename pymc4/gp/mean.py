"""
Mean functions for PyMC4's Gaussian Process Module.

"""
from typing import Union

import tensorflow as tf

from .util import ArrayLike, TfTensor

__all__ = ["Zero", "Constant"]


class Mean:
    r"""Base Class for all the mean functions in GP."""

    def __init__(self, feature_ndims: int = 1):
        self.feature_ndims = feature_ndims

    def __call__(self, X: ArrayLike) -> TfTensor:
        raise NotImplementedError("Your mean function should override this method")

    def __add__(self, mean2):
        return MeanAdd(self, mean2)

    def __mul__(self, mean2):
        return MeanProd(self, mean2)


class MeanAdd(Mean):
    r"""Addition of two or more mean functions

    Parameters
    ----------
    mean1 : {callable, pm.Mean}
        First mean function
    mean2 : {callable, pm.Mean}
        Second mean function
    """

    def __init__(self, mean1: Mean, mean2: Mean):
        if mean1.feature_ndims != mean2.feature_ndims:
            raise ValueError("Cannot combine means with different feature_ndims")
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, X: ArrayLike) -> TfTensor:
        return self.mean1(X) + self.mean2(X)


class MeanProd(Mean):
    r"""Product of two or more mean functions

    Parameters
    ----------
    mean1 : {callable, pm.Mean}
        First mean function
    mean2 : {callable, pm.Mean}
        Second mean function
    """

    def __init__(self, mean1: Mean, mean2: Mean):
        if mean1.feature_ndims != mean2.feature_ndims:
            raise ValueError("Cannot combine means with different feature_ndims")
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, X: ArrayLike) -> TfTensor:
        return self.mean1(X) * self.mean2(X)


class Zero(Mean):
    r"""Zero mean

    Parameters
    ----------
    feature_ndims : int, optional
        number of rightmost dims to include in mean computation
    """

    def __call__(self, X: ArrayLike) -> TfTensor:
        X = tf.convert_to_tensor(X)
        return tf.zeros(X.shape[: -self.feature_ndims])


class Constant(Mean):
    r"""Constant mean

    Parameters
    ----------
    coef : array_like, optional
        co-efficient to scale the mean
    feature_ndims : int, optional
        number of rightmost dims to include in mean computation
    """

    def __init__(self, coef: Union[ArrayLike, float] = 1.0, feature_ndims: int = 1):
        self.coef = coef
        super().__init__(feature_ndims=feature_ndims)

    def __call__(self, X: ArrayLike) -> TfTensor:
        X = tf.convert_to_tensor(X)
        return tf.ones(X.shape[: -self.feature_ndims]) * self.coef
