"""
Covariance Functions for PyMC4's Gaussian Process module.

"""

from abc import abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    #     "Constant",
    #     "WhiteNoise",
    "ExpQuad",
    #     "RatQuad",
    #     "Exponential",
    #     "Matern52",
    #     "Matern32",
    #     "Linear",
    #     "Polynomial",
    #     "Cosine",
    #     "Periodic",
    #     "WarpedInput",
    #     "Gibbs",
    #     "Coregion",
    #     "ScaledCov",
    #     "Kron",
]


class Covariance:
    """Base class of all Covariance functions for Gaussian Process"""

    def __init__(self, feature_ndims, diag=False, **kwargs):
        self.feature_ndims = feature_ndims
        self.diag = diag
        self._kernel = self._init_kernel(feature_ndims=self.feature_ndims, **kwargs)

    @abstractmethod
    def _init_kernel(self, feature_ndims, **kwargs):
        raise NotImplementedError("Your Covariance class should override this method")

    def __call__(self, X1, X2, **kwargs):
        """TODO: docs"""
        if self.diag:
            return tf.linalg.diag_part(self._kernel.apply(X1, X2, **kwargs))
        else:
            return self._kernel.matrix(X1, X2, **kwargs)

    def evaluate_kernel(self, X1, X2, **kwargs):
        return self._kernel.apply(X1, X2, **kwargs)

    def __add__(self, cov2):
        return CovarianceAdd(self, cov2)

    def __mul__(self, cov2):
        return CovarianceProd(self, cov2)

    @property
    def batch_shape(self):
        return self._kernel.batch_shape


class Combination(Covariance):
    """TODO: docs"""

    def __init__(self, cov1, cov2, **kwargs):
        """TODO: docs"""
        self.kernel1 = cov1._kernel
        self.kernel2 = cov2._kernel
        self.feature_ndims = self.kernel1.feature_ndims
        if self.kernel1.feature_ndims != self.kernel2.feature_ndims:
            raise ValueError("Cannot combine kernels with different feature_ndims")
        super().__init__(self.kernel1.feature_ndims, diag=(cov1.diag & cov2.diag), **kwargs)


class CovarianceAdd(Combination):
    """TODO: docs"""

    def _init_kernel(self, feature_ndims, **kwargs):
        """TODO: docs"""
        # TODO: handle the ``diag`` parameter for each kernel being combined
        # Approches: 1. Add the diag support to tfp and use its interface to combine kernels.
        #            2. Create a interface similar to PyMC3 to combine kernels.
        # Currently, I have just ignored the diag parameter.
        return self.kernel1 + self.kernel2


class CovarianceProd(Combination):
    """TODO: docs"""

    def _init_kernel(self, feature_ndims, **kwargs):
        # TODO: Similar problem as CovarianceAdd
        return self.kernel1 * self.kernel2


class Stationary(Covariance):
    """Base class for all Stationary Convarience functions"""

    @property
    def length_scale(self):
        return self._length_scale


class ExpQuad(Stationary):
    """Exponentiated Quadratic Covariance Function
    TODO: docs"""

    def __init__(self, amplitude, length_scale, feature_ndims, diag=False, **kwargs):
        self._amplitude = amplitude
        self._length_scale = length_scale
        super().__init__(feature_ndims=feature_ndims, diag=diag, **kwargs)

    def _init_kernel(self, feature_ndims, **kwargs):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=self._length_scale, amplitude=self._amplitude, feature_ndims=feature_ndims
        )

    @property
    def amplitude(self):
        return self._amplitude
