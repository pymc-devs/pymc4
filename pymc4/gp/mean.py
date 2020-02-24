import tensorflow as tf

__all__ = ["Zero", "Constant"]


class Mean:
    """Base Class for all the mean functions in GP."""

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
        return tf.zeros(X.shape[:-1])


class Constant(Mean):
    """Constant mean"""

    def __init__(self, coef=1.):
        self.coef = coef

    def __call__(self, X):
        X = tf.convert_to_tensor(X)
        return tf.ones(X.shape[:-1])*coef