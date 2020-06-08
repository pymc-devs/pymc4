"""Covariance Functions for PyMC4's Gaussian Process module."""

from typing import Union, Optional
from collections.abc import Iterable
from numbers import Number
from abc import abstractmethod
import operator
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.math.psd_kernels.internal import util

from ._kernel import _Constant, _WhiteNoise
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
    Base class of all Covariance functions for Gaussian Process.

    Parameters
    ----------
    feature_ndims : int
        The number of dimensions to consider as features
        which will be absorbed during the computation.
    active_dims : {int, Iterable}
        A list of (list of) numbers of dimensions in each `feature_ndims`
        columns to operate on. If `None`, defaults to using all the dimensions
        of each `feature_ndims` column. If a single integer `n` is present at `i'th`
        entry of the list, the leftmost `n` dimensions of `i'th` `feature_ndims` column
        are considered for evaluation.
    scale_diag : {Number, array_like}
        Scaling parameter of the lenght_scale parameter of stationary kernels for
        performing Automatic Relevence Detection (ARD).
        Ignored if keyword argument ARD=False.

    Other Parameters
    ----------------
    **kwargs :
        Keyword arguments to pass to the `_init_kernel` method.

    Notes
    -----
    ARD (automatic relevence detection) is done if the length_scale
    is a vector or a tensor. To disable this behaviour, a keyword argument
    `ARD=False` needs to be passed.
    """

    def __init__(
        self,
        feature_ndims: int,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        if not isinstance(feature_ndims, int) or feature_ndims <= 0:
            raise ValueError(
                "expected 'feature_ndims' to be an integer greater or equal to 1"
                f" but found {feature_ndims}."
            )
        if active_dims is not None:
            noslice = slice(None, None)
            self._list_slices = []
            self._slices = []
            if isinstance(active_dims, int):
                active_dims = (active_dims,)
            elif isinstance(active_dims, Iterable):
                active_dims = tuple(active_dims)
            if len(active_dims) > feature_ndims:
                raise ValueError(
                    "'active_dims' contain more entries than number of feature dimensions."
                    " Consider increasing the 'feature_ndims' or decreasing the entries"
                    " in 'active_dims'. expected len(active_dims) < feature_ndims but got"
                    f" {len(active_dims)} > {feature_ndims}"
                )
            active_dims = active_dims + (noslice,) * (feature_ndims - len(active_dims))
            for dim in active_dims:
                if isinstance(dim, Iterable):
                    self._list_slices.append(dim)
                    self._slices.append(noslice)
                else:
                    if not isinstance(dim, slice):
                        self._slices.append(slice(0, dim))
                    else:
                        self._slices.append(dim)
                    self._list_slices.append(noslice)
        self._feature_ndims = feature_ndims
        self._active_dims = active_dims
        self._scale_diag = scale_diag
        ard = kwargs.pop("ARD", True)
        # Initialize Kernel.
        self._kernel = self._init_kernel(feature_ndims=self.feature_ndims, **kwargs)
        # Wrap the kernel in FeatureScaled kernel for ARD.
        if ard:
            if self._scale_diag is None:
                self._scale_diag = 1.0
            self._kernel = tfp.math.psd_kernels.FeatureScaled(
                self._kernel, scale_diag=self._scale_diag
            )

    @abstractmethod
    def _init_kernel(
        self, feature_ndims: int, **kwargs
    ) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        raise NotImplementedError("Your Covariance class should override this method.")

    def _slice(self, X1: TfTensor, X2: TfTensor) -> TfTensor:
        if self._active_dims is None:
            return X1, X2
        # We slice the tensors.
        X1 = X1[..., (*self._slices)]
        X2 = X2[..., (*self._slices)]
        # Workaround for list indices as tensorflow doesn't allow
        # lists as index like numpy. It is not very efficient as
        # `tf.stack` creates a copy instead of view.
        for ax, l in enumerate(self._list_slices):
            if l != slice(None, None):
                rem = (slice(None, None),) * (self._feature_ndims - ax - 1)
                axisX1 = X1.ndim - self._feature_ndims + ax
                axisX2 = X2.ndim - self._feature_ndims + ax
                X1 = tf.stack([X1[..., i, (*rem)] for i in l], axis=axisX1)
                X2 = tf.stack([X2[..., i, (*rem)] for i in l], axis=axisX2)
        return X1, X2

    def _diag(self, X1: ArrayLike, X2: ArrayLike, to_dense=True) -> ArrayLike:
        """Evaluate diagonal part of the full covariance matrix."""
        cov = self(X1, X2)
        if to_dense:
            return cov
        return tf.linalg.diag_part(cov)

    def evaluate_kernel(self, X1: ArrayLike, X2: ArrayLike, **kwargs) -> TfTensor:
        r"""
        Evaluate kernel at certain points.

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

    @property
    def feature_ndims(self) -> int:
        r"""`feature_ndims` parameter of the kernel."""
        return self._feature_ndims

    @property
    def active_dims(self) -> Optional[Union[int, tuple]]:
        r"""Active dimensions of the kernel."""
        return self._active_dims

    @property
    def scale_diag(self) -> Union[Number, ArrayLike]:
        """Scaling parameter of length scale for performing ARD."""
        return self._scale_diag

    def __call__(
        self, X1: ArrayLike, X2: ArrayLike, diag=False, to_dense=True, **kwargs
    ) -> TfTensor:
        r"""
        Evaluate the covariance matrix between the points X1 and X2.

        Parameters
        ----------
        X1 : (..., feature_ndims) array_like
            A tensor of points.
        X2 : (..., feature_ndims) array_like
            A tensor of other points.
        diag : bool, optional
            If true, only evaulates the diagonal of the full covariance matrix.
            (default=False)
        to_dense : bool, optional
            If True, returns full covariance matrix with non-diagonal entries zero
            when `diag=True`. Otherwise, only the diagonal component of the matrix
            is returned. Ignored if `diag=False`. (default=True)

        Other Parameters
        ----------------
        **kwargs : optional
            Keyword arguments to be passed to the `matrix` method of the underlying
            tfp's PSD kernels.

        Returns
        -------
        cov : tensorflow.Tensor
            A covariance matrix with the last `feature_ndims`
            dimensions absorbed to compute the covariance.
        """
        dtyp = util.maybe_get_common_dtype([X1, X2])
        X1 = tf.convert_to_tensor(X1, dtype=dtyp)
        X2 = tf.convert_to_tensor(X2, dtype=dtyp)
        X1, X2 = self._slice(X1, X2)
        if diag:
            return self._diag(X1, X2, to_dense=to_dense)
        return self._kernel.matrix(X1, X2, **kwargs)

    def __add__(self, cov2):
        return _Add(self, cov2)

    def __mul__(self, cov2):
        return _Prod(self, cov2)

    __radd__ = __add__

    __rmul__ = __mul__

    def __array_wrap__(self, result: np.ndarray) -> TfTensor:
        # we retain the original shape to reshape the result later.
        original_shape = result.shape
        # we flatten the array and re-build the left array
        # using the ``.cov2`` attribute of combined kernels.
        result = result.ravel()
        left_array = np.zeros(result.shape, dtype=np.float32)
        for i in range(result.size):
            left_array[i] = result[i].factors[1]
        # reshape the array to its original shape
        left_array = left_array.reshape(original_shape)
        # now, we can put the left array on the right side
        # to create the final combination.
        if isinstance(result[0], _Add):
            return result[0].factors[0] + left_array
        elif isinstance(result[0], _Prod):
            return result[0].factors[0] * left_array


class Combination(Covariance):
    r"""
    Combination of two or more covariance functions.

    Parameters
    ----------
    cov1, cov2, ... : {Covariance, array_like}
        Covariance functions or tensors to be combined to form
        a new covariance function.
    """

    def __init__(self, *factors):
        self.factors = []
        for factor in factors:
            if isinstance(factor, self.__class__):
                self.factors.extend(factor.factors)
            else:
                self.factors.append(factor)
        self._feature_ndims = [
            factor._feature_ndims if isinstance(factor, Covariance) else None for factor in factors
        ]
        self._active_dims = [
            factor._active_dims if isinstance(factor, Covariance) else None for factor in factors
        ]

    def merge_factors(self, X1, X2, diag=False, to_dense=True):
        eval_factors = []
        for factor in self.factors:
            if isinstance(factor, Covariance):
                eval_factors.append(factor(X1, X2, diag=diag, to_dense=to_dense))
            else:
                if diag:
                    eval_factors.append(tf.linalg.diag(tf.linalg.diag_part(factor)))
                else:
                    eval_factors.append(factor)
        return eval_factors


class _Add(Combination):
    r"""
    Addition of two or more covariance functions.

    Parameters
    ----------
    cov1, cov2, ... : {Covariance, array_like}
        Covariance functions or tensors to be combined to form
        a new covariance function.
    """

    @_inherit_docs(Covariance.__call__)
    def __call__(
        self, X1: ArrayLike, X2: ArrayLike, diag=False, to_dense=True, **kwargs
    ) -> TfTensor:
        return reduce(operator.add, self.merge_factors(X1, X2, diag=diag, to_dense=to_dense))


class _Prod(Combination):
    r"""
    Product of two or more covariance functions.

    Parameters
    ----------
    cov1, cov2, ... : {Covariance, array_like}
        Covariance functions or tensors to be combined to form
        a new covariance function.
    """

    @_inherit_docs(Covariance.__call__)
    def __call__(
        self, X1: ArrayLike, X2: ArrayLike, diag=False, to_dense=True, **kwargs
    ) -> TfTensor:
        return reduce(operator.mul, self.merge_factors(X1, X2, diag=diag, to_dense=to_dense))


class Stationary(Covariance):
    r"""Base class for all Stationary Covariance functions."""

    @property
    def length_scale(self) -> Union[ArrayLike, float]:
        r"""Length scale of the covariance function."""
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
    active_dims : {int, Iterable}
        A list of (list of) numbers of dimensions in each `feature_ndims`
        columns to operate on. If `None`, defaults to using all the dimensions
        of each `feature_ndims` column. If a single integer `n` is present at `i'th`
        entry of the list, the leftmost `n` dimensions of `i'th` `feature_ndims` column
        are considered for evaluation.
    scale_diag : {Number, array_like}
        Scaling parameter of the lenght_scale parameter of stationary kernels for
        performing Automatic Relevence Detection (ARD).
        Ignored if keyword argument ARD=False.

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
        feature_ndims: int = 1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._amplitude = amplitude
        self._length_scale = length_scale
        super(ExpQuad, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs
        )

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
        r"""Amplitude of the kernel function."""
        return self._amplitude


class Constant(Stationary):
    r"""
    A Constant Stationary Covariance Function.

    Constant kernel just evaluates to a constant value in each entry of the covariance
    matrix and point evaluations irrespective of the input. It is very useful as a
    lightweight kernel when speed and performance is a primary goal. It doesn’t evaluate
    a complex function and so its gradients are faster and easier to compute.

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
    active_dims : {int, Iterable}
        A list of (list of) numbers of dimensions in each `feature_ndims`
        columns to operate on. If `None`, defaults to using all the dimensions
        of each `feature_ndims` column. If a single integer `n` is present at `i'th`
        entry of the list, the leftmost `n` dimensions of `i'th` `feature_ndims` column
        are considered for evaluation.
    scale_diag : {Number, array_like}
        Scaling parameter of the lenght_scale parameter of stationary kernels for
        performing Automatic Relevence Detection (ARD).
        Ignored if keyword argument ARD=False.

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

    def __init__(
        self,
        coef: Union[float, ArrayLike],
        feature_ndims=1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._coef = coef
        super(Constant, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs
        )

    def _init_kernel(self, feature_ndims, **kwargs):
        return _Constant(self._coef, self._feature_ndims, **kwargs)

    @property
    def coef(self):
        return self._coef


class WhiteNoise(Stationary):
    r"""
    White-noise kernel function.

    This kernel adds some noise to the covariance functions and is mostly
    used to stabilize other PSD kernels. This helps them become non-singular
    and makes cholesky decomposition possible for sampling from the
    `MvNormalCholesky` distribution. It is recommended to use this kernel in
    combination with other covariance/kernel function when working with GP on
    large data.

    .. math::
        k(x_i, x_j') = 1 \text{ if } i = j, 0 \text{ otherwise}

    Parameters
    ----------
    noise : array_like
        The `noise_level` of the kernel.
    feature_ndims : int
        The number of rightmost dimensions to be absorbed during
        the computation or evaluation of the covariance function.
    active_dims : {int, Iterable}
        A list of (list of) numbers of dimensions in each `feature_ndims`
        columns to operate on. If `None`, defaults to using all the dimensions
        of each `feature_ndims` column. If a single integer `n` is present at `i'th`
        entry of the list, the leftmost `n` dimensions of `i'th` `feature_ndims` column
        are considered for evaluation.
    scale_diag : {Number, array_like}
        Scaling parameter of the lenght_scale parameter of stationary kernels for
        performing Automatic Relevence Detection (ARD).
        Ignored if keyword argument ARD=False.

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

    def __init__(
        self,
        noise: Union[float, ArrayLike],
        feature_ndims=1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._noise = noise
        super(WhiteNoise, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs
        )

    def _init_kernel(self, feature_ndims, **kwargs):
        return _WhiteNoise(self._noise, self._feature_ndims, **kwargs)

    @property
    def noise(self):
        return self._noise
