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
    r"""Base class of all Covariance functions for Gaussian Process"""

    def __init__(self, feature_ndims=1, **kwargs):
        # TODO: I have removed the diag parameter
        # for now but I can come back and add it later.
        self.feature_ndims = feature_ndims
        self._kernel = self._init_kernel(feature_ndims=self.feature_ndims, **kwargs)

    @abstractmethod
    def _init_kernel(self, feature_ndims, **kwargs):
        raise NotImplementedError("Your Covariance class should override this method")

    def __call__(self, X1, X2, **kwargs):
        return self._kernel.matrix(X1, X2, **kwargs)

    def evaluate_kernel(self, X1, X2, **kwargs):
        """Evaluate kernel at certain points
        """
        return self._kernel.apply(X1, X2, **kwargs)

    def __add__(self, cov2):
        return CovarianceAdd(self, cov2)

    def __mul__(self, cov2):
        return CovarianceProd(self, cov2)

    @property
    def batch_shape(self):
        return self._kernel.batch_shape


class Combination(Covariance):
    r"""Combination of two or more covariance functions

    Parameters
    ----------
    cov1 : pm.Covariance
        First covariance function.
    cov2 : pm.Covariance
        Second covariance function.
    """

    def __init__(self, cov1, cov2, **kwargs):
        self.kernel1 = cov1._kernel
        self.kernel2 = cov2._kernel
        self.feature_ndims = self.kernel1.feature_ndims
        if self.kernel1.feature_ndims != self.kernel2.feature_ndims:
            raise ValueError("Cannot combine kernels with different feature_ndims")
        super().__init__(self.kernel1.feature_ndims, **kwargs)


class CovarianceAdd(Combination):
    r"""Addition of two or more covariance functions.

    Parameters
    ----------
    feature_ndims : int
        number of rightmost dims to include in kernel computation
    """

    def _init_kernel(self, feature_ndims, **kwargs):
        return self.kernel1 + self.kernel2


class CovarianceProd(Combination):
    r"""Product of two or more covariance functions.

    Parameters
    ----------
    feature_ndims : int
        number of rightmost dims to include in kernel computation
    """

    def _init_kernel(self, feature_ndims, **kwargs):
        return self.kernel1 * self.kernel2


class Stationary(Covariance):
    r"""Base class for all Stationary Covariance functions"""

    @property
    def length_scale(self):
        r"""Length scale of the covariance function"""
        return self._length_scale


class ExpQuad(Stationary):
    r"""Exponentiated Quadratic Stationary Covariance Function.
    Commonly known as the Radial Basis Kernel Function.

    .. math::
       k(x, x') = \sigma^2 \mathrm{exp}\left[ -\frac{(x - x')^2}{2 l^2} \right]

    where :math:`\sigma` = ``amplitude``
          :math:`l` = ``length_scale``

    Parameters
    ----------
    amplitude : tensor, array-like
        The :math:`\sigma` parameter of RBF kernel, amplitude > 0
    length_scale : tensor, array-like
        The :math:`l` parameter of the RBF kernel
    feature_ndims : int, optional
        number of rightmost dims to include in kernel computation
    kwargs : optional
        Other keyword arguments that tfp's ``ExponentiatedQuadratic`` kernel takes
    """

    def __init__(self, amplitude, length_scale, feature_ndims=1, **kwargs):
        if not tf.reduce_all(amplitude > 0.0):
            raise ValueError("`amplitude` parameter must not contains zero or negative entries.")
        self._amplitude = amplitude
        self._length_scale = length_scale
        super().__init__(feature_ndims=feature_ndims, **kwargs)

    def _init_kernel(self, feature_ndims, **kwargs):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=self._length_scale, amplitude=self._amplitude, feature_ndims=feature_ndims
        )

    @property
    def amplitude(self):
        r"""Amplitude of the kernel function"""
        return self._amplitude
