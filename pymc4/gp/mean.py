"""
Mean functions for PyMC4's Gaussian Process Module.

"""

import tensorflow as tf

__all__ = ["Zero", "Constant"]


class Mean:
    """Base Class for all the mean functions in GP."""

    def __init__(self, feature_ndims=1):
        self.feature_ndims = feature_ndims

    def __call__(self, X):
        raise NotImplementedError("Your mean function should override this method")

    def __add__(self, mean2):
        return MeanAdd(self, mean2)

    def __mul__(self, mean2):
        return MeanProd(self, mean2)


class MeanAdd(Mean):
    def __init__(self, mean1, mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, X):
        return self.mean1(X) + self.mean2(X)


class MeanProd(Mean):
    def __init__(self, mean1, mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, X):
        return self.mean1(X) * self.mean2(X)


class Zero(Mean):
    """Zero mean"""

    def __call__(self, X):
        X = tf.convert_to_tensor(X)
        return tf.zeros(X.shape[: -self.feature_ndims])


class Constant(Mean):
    """Constant mean"""

    def __init__(self, coef=1.0, feature_ndims=1):
        self.coef = coef
        super().__init__(feature_ndims=feature_ndims)

    def __call__(self, X):
        X = tf.convert_to_tensor(X)
        return tf.ones(X.shape[: -self.feature_ndims]) * coef
