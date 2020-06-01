"""
Covariance Functions for PyMC4's Gaussian Process module.

"""
from typing import Union
from abc import abstractmethod

import numpy as np
import tensorflow_probability as tfp

from .kernel import _Constant, _WhiteNoise
from .util import ArrayLike, TfTensor, _inherit_docs


__all__ = [
    "Constant",
    "WhiteNoise",
    "ExpQuad",
    # "RatQuad",
    # "Exponential",
    # "Matern52",
    # "Matern32",
    # "Linear",
    # "Polynomial",
    # "Cosine",
    # "Periodic",
    # "WarpedInput",
    # "Gibbs",
    # "Coregion",
    # "ScaledCov",
    # "Kron",
]


class Covariance:
    r"""
    Base class of all Covariance functions for Gaussian Process

    Parameters
    ----------
    feature_ndims : int
        The number of dimensions to consider as features
        which will be absorbed during the computation.

    Other Parameters
    ----------------
    **kwargs :
        Keyword arguments to pass to the `_init_kernel` method

    Notes
    -----
    ARD (automatic relevence detection) is done if the length_scale
    is a vector or a tensor. To disable this behaviour, a keyword argument
    `ARD=False` needs to be passed.
    """

    def __init__(self, feature_ndims: int, **kwargs):
        # TODO: Implement the `diag` parameter as in PyMC3.
        self._feature_ndims = feature_ndims
        self._kernel = self._init_kernel(feature_ndims=self.feature_ndims, **kwargs)
        if self._kernel is not None:
            # wrap the kernel in FeatureScaled kernel for ARD
            if kwargs.pop("ARD", True):
                self._scale_diag = kwargs.pop("scale_diag", 1.0)
                self._kernel = tfp.math.psd_kernels.FeatureScaled(
                    self._kernel, scale_diag=self._scale_diag
                )

    @abstractmethod
    def _init_kernel(
        self, feature_ndims: int, **kwargs
    ) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        raise NotImplementedError("Your Covariance class should override this method")

    def __call__(self, X1: ArrayLike, X2: ArrayLike, **kwargs) -> TfTensor:
        r"""
        Evaluate the covariance matrix between the points X1 and X2.

        Parameters
        ----------
        X1 : (..., feature_ndims) array_like
            A tensor of points.
        X2 : (..., feature_ndims) array_like
            A tensor of other points.

        Returns
        -------
        cov : tensorflow.Tensor
            A covariance matrix with the last `feature_ndims`
            dimensions absorbed to compute the covariance.
        """
        return self._kernel.matrix(X1, X2, **kwargs)

    def evaluate_kernel(self, X1: ArrayLike, X2: ArrayLike, **kwargs) -> TfTensor:
        r"""
        Evaluate kernel at certain points

        Parameters
        ----------
        X1 : array_like
            First point(s)
        X2 : array_like
            Second point(s)

        Returns
        -------
        cov : tensorflow.Tensor
            Covariance between pair of points in `X1` and `X2`.
        """
        return self._kernel.apply(X1, X2, **kwargs)

    def __add__(self, cov2):
        return CovarianceAdd(self, cov2)

    def __mul__(self, cov2):
        return CovarianceProd(self, cov2)

    __radd__ = __add__

    __rmul__ = __mul__

    def __array_wrap__(self, result: np.ndarray) -> TfTensor:
        # we retain the original shape to reshape the result later
        original_shape = result.shape
        # we flatten the array and re-build the left array
        # using the ``.cov2`` attribute of combined kernels.
        result = result.ravel()
        left_array = np.zeros_like(result)
        for i in range(result.size):
            left_array[i] = result[i].cov2
        # reshape the array to its original shape
        left_array = left_array.reshape(original_shape)
        # now, we can put the left array on the right side
        # to create the final combination.
        if isinstance(result[0], CovarianceAdd):
            return result[0] + left_array
        elif isinstance(result[0], CovarianceProd):
            return result[0] * left_array

    @property
    def feature_ndims(self) -> int:
        f"""Returns the `feature_ndims` of the kernel"""
        return self._feature_ndims


class Combination(Covariance):
    r"""
    Combination of two or more covariance functions

    Parameters
    ----------
    cov1 : pm.gp.Covariance
        First covariance function.
    cov2 : pm.gp.Covariance
        Second covariance function.
    """

    def __init__(self, cov1: Union[Covariance, ArrayLike], cov2: Union[Covariance, ArrayLike]):
        self.cov1 = cov1
        self.cov2 = cov2
        if isinstance(cov1, Covariance) and isinstance(cov2, Covariance):
            if cov1.feature_ndims != cov2.feature_ndims:
                raise ValueError("Cannot combine kernels with different feature_ndims")

    @property
    def feature_ndims(self) -> int:
        if isinstance(self.cov1, Covariance):
            return self.cov1.feature_ndims
        return self.cov2.feature_ndims


class CovarianceAdd(Combination):
    r"""
    Addition of two or more covariance functions.

    Parameters
    ----------
    feature_ndims : int
        The number of rightmost dimensions to be absorbed during
        the computation or evaluation of the covariance function.

    Other Parameters
    ----------------
    **kwargs:
        Keyword arguments to pass to the covariance `metrix` method
    """

    @_inherit_docs(Covariance.__call__)
    def __call__(self, X1: ArrayLike, X2: ArrayLike, **kwargs) -> TfTensor:
        if not isinstance(self.cov1, Covariance):
            return self.cov1 + self.cov2(X1, X2, **kwargs)
        elif not isinstance(self.cov2, Covariance):
            return self.cov2 + self.cov1(X1, X2, **kwargs)
        else:
            return self.cov1(X1, X2, **kwargs) + self.cov2(X1, X2, **kwargs)


class CovarianceProd(Combination):
    r"""
    Product of two or more covariance functions.

    Parameters
    ----------
    feature_ndims : int
        The number of rightmost dimensions to be absorbed during
        the computation or evaluation of the covariance function.

    Other Parameters
    ----------------
    **kwargs:
        Keyword arguments to pass to the `_init_kernel` method
    """

    @_inherit_docs(Covariance.__call__)
    def __call__(self, X1: ArrayLike, X2: ArrayLike, **kwargs) -> TfTensor:
        if not isinstance(self.cov1, Covariance):
            return self.cov1 * self.cov2(X1, X2, **kwargs)
        elif not isinstance(self.cov2, Covariance):
            return self.cov2 * self.cov1(X1, X2, **kwargs)
        else:
            return self.cov1(X1, X2, **kwargs) * self.cov2(X1, X2, **kwargs)


class Stationary(Covariance):
    r"""Base class for all Stationary Covariance functions"""

    @property
    def length_scale(self) -> Union[ArrayLike, float]:
        r"""Length scale of the covariance function"""
        return self._length_scale  # type: ignore


class ExpQuad(Stationary):
    r"""
    Exponentiated Quadratic Stationary Covariance Function.
    A kernel from the Radial Basis family of kernels.

    .. math::
       k(x, x') = \sigma^2 \mathrm{exp}\left[ -\frac{(x - x')^2}{2 l^2} \right]

    where :math:`\sigma` = ``amplitude``
          :math:`l` = ``length_scale``

    Parameters
    ----------
    amplitude : array_like
        The :math:`\sigma` parameter of RBF kernel, amplitude > 0
    length_scale : array_like
        The :math:`l` parameter of the RBF kernel
    feature_ndims : int
        The number of rightmost dimensions to be absorbed during
        the computation or evaluation of the covariance function.

    Other Parameters
    ----------------
    **kwargs :
        Other keyword arguments that tfp's ``ExponentiatedQuadratic`` kernel takes

    Examples
    --------
    >>> from pymc4.gp.cov import ExpQuad
    >>> import numpy as np
    >>> X1 = np.array([[1.], [2.], [3.]])
    >>> X2 = np.array([[3.], [2.], [1.]])
    >>> kernel = ExpQuad(amplitude=1., length_scale=1., feature_ndims=1)
    >>> kernel(X1, X2)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[0.13533528, 0.60653067, 1.        ],
        [0.60653067, 1.        , 0.60653067],
        [1.        , 0.60653067, 0.13533528]], dtype=float32)>
    >>> kernel.evaluate_kernel(X1, X2)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.13533528, 1.        , 0.13533528], dtype=float32)>

    Notes
    -----
    ARD (automatic relevence detection) is done if the length_scale
    is a vector or a tensor. To disable this behaviour, a keyword argument
    `ARD=False` needs to be passed.
    """

    def __init__(
        self,
        length_scale: Union[ArrayLike, float],
        amplitude: Union[ArrayLike, float] = 1.0,
        feature_ndims=1,
        **kwargs,
    ):
        self._amplitude = amplitude
        self._length_scale = length_scale
        super().__init__(feature_ndims=feature_ndims, **kwargs)

    def _init_kernel(
        self, feature_ndims: int, **kwargs
    ) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function"""
        return self._amplitude


class Constant(Stationary):
    r"""
    A Constant Stationary Covariance Function.

    .. math::
        k(x, x') = c

    where :math:`c` = :code:`coef`

    Parameters
    ----------
    coef : array_like
        The constant coefficient indicating the covariance
        between any two points. It is the constant `c` in
        the equation above.
    feature_ndims : int
        The number of rightmost dimensions to be absorbed during
        the computation or evaluation of the covariance function.

    Other Parameters
    ----------------
    **kwargs:
        Keyword arguments to pass to the `_Constant` kernel.

    Examples
    --------
    >>> from pymc4.gp.cov import Constant
    >>> import numpy as np
    >>> k = Constant(coef=5., feature_ndims=1)
    >>> k
    <pymc4.gp.cov.Constant object at 0x000001C96936DE10>
    >>> X1 = np.array([[1.], [2.], [3.]])
    >>> X2 = np.array([[4.], [5.], [6.]])
    >>> k(X1, X2)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[5., 5., 5.],
        [5., 5., 5.],
        [5., 5., 5.]], dtype=float32)>

    Notes
    -----
    ARD (automatic relevence detection) is done if the length_scale
    is a vector or a tensor. To disable this behaviour, a keyword argument
    `ARD=False` needs to be passed.
    """

    def __init__(self, coef: Union[float, ArrayLike], feature_ndims=1, **kwargs):
        self._coef = coef
        super().__init__(feature_ndims=feature_ndims, **kwargs)

    def _init_kernel(self, feature_ndims, **kwargs):
        return _Constant(self._coef, self._feature_ndims, **kwargs)


class WhiteNoise(Stationary):
    r"""
    White-noise kernel function. This kernel adds some noise
    to the covariance functions and is mostly used to stabilize
    other PSD kernels. This helps them become non-singular and makes
    cholesky decomposition possible for sampling from the MvNormalCholesky.
    It is recommended to use this kernel in combination with other
    covariance/kernel function when working with GP on large data.

    .. math::
        k(x_i, x_j') = 1 \text{ if } i = j, 0 \text{ otherwise}

    Parameters
    ----------
    noise : array_like
        The `noise_level` of the kernel.
    feature_ndims : int
        The number of rightmost dimensions to be absorbed during
        the computation or evaluation of the covariance function.

    Other Parameters
    ----------------
    **kwargs :
        Keyword arguments to pass to the `_WhiteNoise` kernel.

    Examples
    --------
    >>> from pymc4.gp.cov import WhiteNoise
    >>> import numpy as np
    >>> k = WhiteNoise(noise=1e-4, feature_ndims=1)
    >>> k
    <pymc4.gp.cov.WhiteNoise object at 0x00000162FC073390>
    >>> X1 = np.array([[1.], [2.]])
    >>> X2 = np.array([[3.], [4.]])
    >>> k(X1, X2)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.e-04, 0.e+00],
        [0.e+00, 1.e-04]], dtype=float32)>

    Notes
    -----
    This kernel function dosn't have a point evaluation scheme.
    Hence, the `pymc4.gp.cov.WhiteNoise.evaluate_kernel` method
    raises a `NotImplementedError` when called.
    """

    def __init__(self, noise: Union[float, ArrayLike], feature_ndims=1, **kwargs):
        self._noise = noise
        super().__init__(feature_ndims=feature_ndims, **kwargs)

    def _init_kernel(self, feature_ndims, **kwargs):
        return _WhiteNoise(self._noise, self._feature_ndims, **kwargs)
