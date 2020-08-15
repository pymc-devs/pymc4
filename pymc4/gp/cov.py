"""Covariance Functions for PyMC4's Gaussian Process module.

Wraps `tensorflow_probability.python.math.psd_kernels` module and provides
many more kernels with all necessary features to work with Gaussian processes.
"""

from typing import Union, Optional, Any, List, Callable
from collections.abc import Iterable
from numbers import Number
from abc import abstractmethod
import operator
from functools import reduce, partial

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.math.psd_kernels import (
    ExponentiatedQuadratic,
    RationalQuadratic,
    MaternOneHalf,
    MaternFiveHalves,
    MaternThreeHalves,
    ExpSinSquared,
    PositiveSemidefiniteKernel,
    FeatureScaled,
    FeatureTransformed,
    Polynomial as TFPPolynomial,
    Linear as TFPLinear,
)

# from tensorflow_probability.python.math.psd_kernels.internal import util

from ._kernel import _Constant, _WhiteNoise, _Exponential, _Gibbs, _Cosine, _ScaledCov
from .util import ArrayLike, TfTensor, _inherit_docs, _build_docs


__all__ = [
    "Covariance",
    "Constant",
    "WhiteNoise",
    "ExpQuad",
    "RatQuad",
    "Exponential",
    "Matern52",
    "Matern32",
    "Matern12",
    "Linear",
    "Polynomial",
    "Cosine",
    "Periodic",
    "WarpedInput",
    "Gibbs",
    # "Coregion",
    "ScaledCov",
    # "Kron",
]

_common_doc = """feature_ndims : int, optional
        The number of dimensions to consider as features which will be absorbed
        during the computation. Defaults to 1. Increasing this causes significant
        overhead in computation. Consider using ``active_dims`` parameter along with
        this parameter for better performance.
    active_dims : {int, Iterable}, optional
        A list of (list of) numbers of dimensions in each ``feature_ndims``
        columns to operate on. If ``None``, defaults to using all the dimensions
        of each ``feature_ndims`` column. If a single integer ``n`` is present at ``i'th``
        entry of the list, the leftmost ``n`` dimensions of ``i'th`` ``feature_ndims`` column
        are considered for evaluation.
    scale_diag : {Number, array_like}, optional
        Scaling parameter of the lenght_scale parameter of stationary kernels for
        performing Automatic Relevance Detection (ARD). Ignored if keyword argument ``ARD=False``."""

_note_doc = """ARD (automatic relevance detection) is performed if the parameter ``length_scale``
    is a vector or a tensor. To disable this behaviour, a keyword argument
    ``ARD=False`` needs to be passed. Other keyword arguments that can be passed are:
    validate_args : bool
        A boolean indicating whether or not to validate arguments. Incurs a little
        overhead when set to ``True``. Its default value is ``False``.
    name:
        You can optionally give a name to the kernels. All the operations will be
        preformed under the name "<name>/<operation_name>:0"."""


@_build_docs
class Covariance:
    r"""
    Base class of all Covariance functions for Gaussian Process.

    Parameters
    ----------
    %(_common_doc)

    Other Parameters
    ----------------
    **kwargs :
        Keyword arguments to pass to the ``_init_kernel`` method.

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self,
        feature_ndims: int,
        active_dims: Optional[Union[int, Iterable]],
        scale_diag: Optional[Union[ArrayLike, Number]],
        **kwargs,
    ):
        if not isinstance(feature_ndims, int) or feature_ndims <= 0:
            raise ValueError(
                "expected 'feature_ndims' to be an integer greater or equal to 1"
                f" but found {feature_ndims}."
            )

        if active_dims is not None:
            noslice = slice(None, None)
            # Tensorflow doesn't allow slicing with list indices.
            # So, ``_list_slices`` holds all the list indices which
            # will be handled differently by the ``_slice`` method.
            self._list_slices: Iterable[Any] = []
            self._slices: Iterable[Any] = []

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
        self._ard = kwargs.pop("ARD", True)
        # Initialize Kernel.
        self._kernel = self._init_kernel(feature_ndims=self.feature_ndims, **kwargs)
        # Wrap the kernel in FeatureScaled kernel for ARD.
        if self._ard:
            if self._scale_diag is None:
                self._scale_diag = tf.constant(1, dtype=self._kernel.dtype)
            self._kernel = FeatureScaled(self._kernel, scale_diag=self._scale_diag)

    @abstractmethod
    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        raise NotImplementedError("Your Covariance class should override this method.")

    def _slice(self, X1: TfTensor, X2: TfTensor) -> TfTensor:
        if self._active_dims is None:
            return X1, X2

        # We first slice the tensors using non-list indices.
        # This is fast and creates a view instead of a copy.
        X1 = X1[..., (*self._slices)]
        X2 = X2[..., (*self._slices)]

        # Workaround for list indices as tensorflow doesn't allow
        # lists as index like numpy. It is not very efficient as
        # ``tf.gather`` creates a copy instead of view.
        for ax, ind in enumerate(self._list_slices):
            if ind != slice(None, None):
                X1 = tf.gather(X1, indices=ind, axis=ax - self._feature_ndims)
                X2 = tf.gather(X2, indices=ind, axis=ax - self._feature_ndims)
        return X1, X2

    def _diag(self, X1: ArrayLike, X2: ArrayLike, to_dense=True) -> ArrayLike:
        """Evaluate only the diagonal part of the full covariance matrix."""
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
            Covariance between each pair of points in ``X1`` and ``X2``.
        """
        return self._kernel.apply(X1, X2, **kwargs)

    @property
    def feature_ndims(self) -> int:
        r"""``feature_ndims`` parameter of the kernel."""
        return self._feature_ndims

    @property
    def active_dims(self) -> Optional[Union[int, tuple]]:
        r"""Active dimensions of the kernel."""
        return self._active_dims

    @property
    def scale_diag(self) -> Union[Number, ArrayLike]:
        r"""Scaling parameter of length scale for performing ARD."""
        return self._scale_diag

    @property
    def ard(self) -> bool:
        r"""Weather ARD is enabled or not."""
        return self._ard

    def __call__(
        self, X1: ArrayLike, X2: ArrayLike, diag=False, to_dense=True, **kwargs
    ) -> TfTensor:
        r"""
        Evaluate the covariance matrix between the points ``X1`` and ``X2``.

        Parameters
        ----------
        X1 : array_like of shape ``(..., feature_ndims)``
            A tensor of points.
        X2 : array_like of shape ``(..., feature_ndims)``
            A tensor of other points.
        diag : bool, optional
            If true, only evaluates the diagonal of the full covariance matrix.
            (default=False)
        to_dense : bool, optional
            If True, returns full covariance matrix with non-diagonal entries zero
            when ``diag=True``. Otherwise, only the diagonal component of the matrix
            is returned. Ignored if ``diag=False``. (default=True)

        Other Parameters
        ----------------
        **kwargs : optional
            Keyword arguments to be passed to the ``matrix`` method of the underlying
            tfp's PSD kernels. Ignored if ``ARD=True``.

        Returns
        -------
        cov : tensorflow.Tensor
            A covariance matrix with the last ``feature_ndims``
            dimensions absorbed to compute the covariance.
        """
        with self._kernel._name_and_control_scope("call"):
            X1 = tf.convert_to_tensor(X1, dtype_hint=self._kernel.dtype)
            X2 = tf.convert_to_tensor(X2, dtype_hint=self._kernel.dtype)
            X1, X2 = self._slice(X1, X2)
            # if self._ard and not diag:
            #     X1 = util.pad_shape_with_ones(X1, ndims=1, start=-(self.feature_ndims + 1))
            #     X2 = util.pad_shape_with_ones(X2, ndims=1, start=-(self.feature_ndims + 2))
            #     try:
            #         return self._kernel._apply(X1, X2, example_ndims=0)
            #     except NotImplementedError:
            #         return self._kernel.matrix(X1, X2, **kwargs)
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
        r"""Combine cov functions with NumPy arrays on the left."""
        original_shape = result.shape
        result = result.ravel()
        left_array = np.zeros(result.shape, dtype=np.float32)

        for i in range(result.size):
            left_array[i] = result[i].factors[1]
        left_array = left_array.reshape(original_shape)

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
        self._feature_ndims = max(
            [factor._feature_ndims if isinstance(factor, Covariance) else 1 for factor in factors]
        )
        self._active_dims = [
            factor._active_dims if isinstance(factor, Covariance) else None for factor in factors
        ]

    def _eval_factor(self, factor, X1, X2, diag=False, to_dense=False):
        if isinstance(factor, Covariance):
            return factor(X1, X2, diag=diag, to_dense=to_dense)
        if diag:
            return tf.linalg.diag(tf.linalg.diag_part(factor))
        else:
            return factor

    def merge_factors(
        self, X1: ArrayLike, X2: ArrayLike, diag=False, to_dense=True
    ) -> List[TfTensor]:
        fn = partial(self._eval_factor, X1=X1, X2=X2, diag=diag, to_dense=to_dense)
        return tf.nest.map_structure(fn, self.factors)


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


_ls_amp_doc = """length_scale : array_like
        The length-scale ℓ determines the length of the 'wiggles' in your function. In general,
        you won't be able to extrapolate more than ℓ units away from your data. If a float,
        an isotropic kernel is used. If an array and ``ARD=True``, an anisotropic kernel
        is used where each dimension defines the length-scale of the respective feature dimension.
    amplitude : array_like, optional
        Amplitude is a scaling factor that determines the average distance of your function away
        from your mean. Every kernel has this parameter out in its front. If a float,
        an isotropic kernel is used. If an array and ``ARD=True``, an anisotropic kernel
        is used where each dimension defines the amplitude of the respective feature dimension.
        (default=1)"""


@_build_docs
class ExpQuad(Stationary):
    r"""
    Exponentiated Quadratic Stationary Covariance Function.

    This is the most used kernel in GP Modelling because of its mathematical properties.
    It is infinitely differentiable and forces the covariance function to be smooth.
    It comes from Squared Exponential (SE) family of kernels which is also commonly called
    as the Radial Basis Kernel (RBF) Family.

    .. math::

       k(x, x') = \sigma^2 \mathrm{exp}\left[ -\frac{(x - x')^2}{2 l^2} \right]

    where :math:`\sigma` = ``amplitude`` and
          :math:`l` = ``length_scale``

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_common_doc)

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
    %(_note_doc)
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
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return ExponentiatedQuadratic(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude


@_build_docs
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
        between any two points. It is the constant ``c`` in
        the equation above.
    %(_common_doc)

    Other Parameters
    ----------------
    **kwargs:
        Keyword arguments to pass to the ``_Constant`` kernel.

    Examples
    --------
    >>> from pymc4.gp.cov import Constant
    >>> import numpy as np
    >>> k = Constant(coef=5., feature_ndims=1)
    >>> X1 = np.array([[1.], [2.], [3.]])
    >>> X2 = np.array([[4.], [5.], [6.]])
    >>> k(X1, X2)
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[5., 5., 5.],
           [5., 5., 5.],
           [5., 5., 5.]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
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
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims, **kwargs):
        return _Constant(self._coef, self._feature_ndims, **kwargs)

    @property
    def coef(self):
        return self._coef


@_build_docs
class WhiteNoise(Stationary):
    r"""
    White-noise kernel function.

    This kernel adds some noise to the covariance functions and is mostly
    used to stabilize other PSD kernels. This helps them become non-singular
    and makes cholesky decomposition possible for sampling from the
    ``MvNormalCholesky`` distribution. It is recommended to use this kernel in
    combination with other covariance/kernel function when working with GP on
    large data.

    .. math::

        k(x_i, x_j') = 1 \text{ if } i = j, 0 \text{ otherwise}

    Parameters
    ----------
    noise : array_like
        The ``noise_level`` of the kernel.
    %(_common_doc)

    Other Parameters
    ----------------
    **kwargs :
        Keyword arguments to pass to the ``_WhiteNoise`` kernel.

    Examples
    --------
    >>> from pymc4.gp.cov import WhiteNoise
    >>> import numpy as np
    >>> k = WhiteNoise(noise=1e-4, feature_ndims=1)
    >>> X1 = np.array([[1.], [2.]])
    >>> X2 = np.array([[3.], [4.]])
    >>> k(X1, X2)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.e-04, 0.e+00],
           [0.e+00, 1.e-04]], dtype=float32)>

    Notes
    -----
    This kernel function dosn't have a point evaluation scheme.
    Hence, the ``pymc4.gp.cov.WhiteNoise.evaluate_kernel`` method
    raises a ``NotImplementedError`` when called.

    %(_note_doc)
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
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims, **kwargs):
        return _WhiteNoise(self._noise, self._feature_ndims, **kwargs)

    @property
    def noise(self):
        return self._noise


@_build_docs
class RatQuad(Stationary):
    r"""
    Rational Quadratic Kernel.

    This kernel belongs to the RBF Family of kernels and is a generalization
    over the ``ExpQuad`` kernel. ``scale_mixtue_rate`` parameter controls the
    mixture of length-scales to use. This kernel becomes equivalent to the
    ``ExpQuad`` kernel when ``scale_mixtue_rate`` approaches infinity.

    .. math::

        k(x, x') = \sigma^2 \left(1 + \frac{\|x-x'\|^2}{2\alpha l^2}\right)^{\alpha}

    where :math:`\alpha` = ``scale_mixture_rate``, :math:`l` = ``length_scale`` and
    :math:`\sigma` = ``amplitude``.

    Parameters
    ----------
    %(_ls_amp_doc)
    scale_mixture_rate: array_like, optional
        The mixture of length-scales to use. Equivalent to adding ``ExpQuad``
        kernels with different ``length_scale``s. When this parameter approaches
        infinity, it becomes equivalent to the ``ExpQuad`` kernel. (default=1)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import RatQuad
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = RatQuad(length_scale=1.)
    >>> k(x, x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.        , 0.19999999],
           [0.19999999, 1.        ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self,
        length_scale: Union[ArrayLike, float],
        amplitude: Union[ArrayLike, float] = 1.0,
        scale_mixture_rate: Union[ArrayLike, float] = 1.0,
        feature_ndims: int = 1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._amplitude = amplitude
        self._length_scale = length_scale
        self._scale_mixture_rate = scale_mixture_rate
        super(RatQuad, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return RationalQuadratic(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            scale_mixture_rate=self._scale_mixture_rate,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude

    @property
    def scale_mixture_rate(self) -> Union[ArrayLike, float]:
        r"""Scale Mixture Rate of the kernel."""
        return self._scale_mixture_rate


_matern_doc = r"""Matern family of kernels is a generalization over the RBF family of kernels.
    The :math:`\\nu` parameter controls the smoothness of the kernel function. Higher
    values of :math:`\\nu` result in a more smooth function. The most common values
    of :math:`\\nu` are 0.5, 1.5 and 2.5 as the modified Bessel's function is analytical
    there."""


@_build_docs
class Matern12(Stationary):
    r"""
    Matern 1/2 kernel.

    %(_matern_doc)

    This kernel has the value of :math:`nu = 0.5`.

    It can be given as:

    .. math::

        k(x, x') = \sigma^2\mathrm{exp}\left( -\frac{\|x - x'\|^2}{\ell} \right)

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Matern12
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Matern12(1.)
    >>> k(x, x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.        , 0.05910575],
           [0.05910575, 1.        ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
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
        super(Matern12, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return MaternOneHalf(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude


@_build_docs
class Matern32(Stationary):
    r"""
    Matern 3/2 kernel.

    %(_matern_doc)

    This kernel has the value of :math:`nu = 1.5`.

    It can be given as:

    .. math::

        k(x, x') = \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)
                   \mathrm{exp}\left( - \frac{\sqrt{3(x - x')^2}}{\ell} \right)

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Matern32
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Matern32(1.)
    >>> k(x, x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.       , 0.0439721],
           [0.0439721, 1.       ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
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
        super(Matern32, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return MaternThreeHalves(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude


@_build_docs
class Matern52(Stationary):
    r"""
    Matern 5/2 kernel.

    %(_matern_doc)

    This kernel has the value of :math:`nu = 2.5`.

    It can be given as:

    .. math::

        k(x, x') = \left(1 + \frac{\sqrt{5(x - x')^2}}{\ell} +
                   \frac{5(x-x')^2}{3\ell^2}\right)
                   \mathrm{exp}\left( - \frac{\sqrt{5(x - x')^2}}{\ell} \right)

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Matern52
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Matern52(1.)
    >>> k(x, x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.        , 0.03701403],
           [0.03701403, 1.        ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
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
        super(Matern52, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return MaternFiveHalves(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude


_linear_doc = """bias_variance : array_like
        The bias to add in the linear equation. This parameter controls
        how far your covariance is from the mean value.
    slope_variance : array_like
        The slope of the linear equation. This parameter controls how fast
        the covariance increases from the origin point.
    shift : array_like
        The amount of shift to apply to normalize the input vectors (or arrays).
        This parameter brings all the input values closer to origin by ``shift`` units."""


@_build_docs
class Linear(Covariance):
    r"""
    Linear Kernel.

    This kernel evaluates a linear function of the inputs :math:`x` and :math:`x'`.
    It performs bayesian linear regression on the inputs to produce random linear functions.

    .. math::

        k(x, x') = \sigma_b^2 + \sigma_v^2\left(x - c\right)\left(x'-c\right)

    where :math:`\sigma_b` = ``bias_variance``, :math:`\sigma_v` = ``slope_variance``,
          and :math:`c` = ``shift``.

    Parameters
    ----------
    %(_linear_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Linear
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Linear(1.5, 2., 1.)
    >>> k(x, x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[ 6.25, 14.25],
           [14.25, 54.25]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self,
        bias_variance: Union[ArrayLike, float],
        slope_variance: Union[ArrayLike, float],
        shift: Union[ArrayLike, float],
        feature_ndims: int = 1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._bias_variance = bias_variance
        self._slope_variance = slope_variance
        self._shift = shift
        super(Linear, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return TFPLinear(
            bias_variance=self._bias_variance,
            slope_variance=self._slope_variance,
            shift=self._shift,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def slope_variance(self) -> Union[ArrayLike, float]:
        r"""``slope_variance`` parameter of the kernel."""
        return self._slope_variance

    @property
    def bias_variance(self) -> Union[ArrayLike, float]:
        r"""``bias_variance`` parameter of the kernel."""
        return self._bias_variance

    @property
    def shift(self) -> Union[ArrayLike, float]:
        r"""``shift`` parameter of the kernel."""
        return self._shift


@_build_docs
class Polynomial(Covariance):
    r"""
    Polynomial Kernel.

    This kernel is a generalization over the linear kernel. An extra term
    ``exponent`` controls the degrees of the underlying polynomial function.
    If the parameter ``slope_variance`` is a vector or a general tensor, ARD
    is performed and each entry in the ``slope_variance`` acts like a co-efficient
    of each feature dimension of the input array. When the ``exponent = 1``, it
    becomes a linear kernel.

    .. math::

        k(x, x') = \sigma_b^2 + \sigma_v^2\left(
                   \left(x - c\right)\left(x'-c\right)\right)^{\alpha}

    where :math:`\sigma_b` = ``bias_variance``, :math:`\sigma_v` = ``slope_variance``,
          :math:`c` = ``shift``, and :math:`\alpha` = ``exponent``.

    Parameters
    ----------
    %(_linear_doc)
    exponent : array_like
        Exponent (degree) of the underlying polynomial function.
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Exponential
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Polynomial(1.5, 2., 1., 2.)
    >>> k(x,x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[  6.25,  38.25],
           [ 38.25, 678.25]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self,
        bias_variance: Union[ArrayLike, float],
        slope_variance: Union[ArrayLike, float],
        shift: Union[ArrayLike, float],
        exponent: Union[ArrayLike, float],
        feature_ndims: int = 1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._bias_variance = bias_variance
        self._slope_variance = slope_variance
        self._shift = shift
        self._exponent = exponent
        super(Polynomial, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return TFPPolynomial(
            bias_variance=self._bias_variance,
            slope_variance=self._slope_variance,
            shift=self._shift,
            exponent=self._exponent,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def slope_variance(self) -> Union[ArrayLike, float]:
        r"""``slope_variance`` parameter of the kernel."""
        return self._slope_variance

    @property
    def bias_variance(self) -> Union[ArrayLike, float]:
        r"""``bias_variance`` parameter of the kernel."""
        return self._bias_variance

    @property
    def shift(self) -> Union[ArrayLike, float]:
        r"""``shift`` parameter of the kernel."""
        return self._shift

    @property
    def exponent(self) -> Union[ArrayLike, float]:
        r"""``exponent`` parameter of the kernel."""
        return self.exponent


_period_doc = """period : array_like, optional
        This parameter defines the period of a periodic kernel. If a float,
        an isotropic kernel is used. If an array and ``ARD=True``, an
        anisotropic kernel is used where each dimension defines the period
        of the respective feature dimension. (default=1)"""


@_build_docs
class Periodic(Covariance):
    r"""
    Periodic aka Exponential Sine Squared Kernel.

    Periodic kernel aka Exponential Sine Squared Kernel comes from the Periodic Family
    of kernels. This kernels occilates in space with a period `T`. This kernel is used
    mostly for time-series and other temporal prediction tasks.

    This kernel can be expressed as:

    .. math::

        k(x, x') = \sigma^2 \exp\left(-\frac{\sin^2\left(\frac{\pi\|x-x'\|^2}
                   {T}\right)}{2\ell^2}\right)

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_period_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Periodic
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Periodic(1., 1., 1.)
    >>> k(x,x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self,
        length_scale: Union[ArrayLike, float],
        amplitude: Union[ArrayLike, float] = 1.0,
        period: Union[ArrayLike, float] = 1.0,
        feature_ndims: int = 1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._amplitude = amplitude
        self._length_scale = length_scale
        self._period = period
        super(Periodic, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return ExpSinSquared(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            period=self._period,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude

    @property
    def period(self) -> Union[ArrayLike, float]:
        return self._period


@_build_docs
class Exponential(Stationary):
    r"""Exponential Kernel.

    This kernel is used as an alternative to the ``ExpQuad`` kernel.
    It is also known as Laplacian Kernel.

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Exponential
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Exponential(1., 1.)
    >>> k(x,x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[0.9999995 , 0.24311674],
           [0.24311674, 0.9999995 ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
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
        super(Exponential, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return _Exponential(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude


@_build_docs
class Gibbs(Covariance):
    r"""
    Gibbs Non-Stationary kernel.

    This kernel uses length-scales that are a function of the input.
    Hence, this comes from the family of non-stationary kernels. It
    is computationally expensive but provides a very flexible function
    , with which very complex data can be modeled easily.

    .. math::

        k(x, x') = \sqrt{\frac{2\ell(x)\ell(x')}{\ell^2(x) + \ell^2(x')}}
                   \mathrm{exp}\left( -\frac{(x - x')^2}
                   {\ell^2(x) + \ell^2(x')} \right)

    Parameters
    ----------
    length_scale_fn : callable
        This is a function of the inputs which outputs a ``array_like``
        object which are the length-scales to be used for that particular
        input. The output must have the same shape or a broadcastable shape
        with the input.
    fn_args : tuple, optional
        A tuple of other arguments to be passed to the function. If None,
        defaults to passing no extra arguments.
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Gibbs
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Gibbs(lambda x: tf.ones(x.shape))
    >>> k(x,x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.        , 0.01831564],
           [0.01831564, 1.        ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self,
        length_scale_fn: Callable,
        fn_args: Optional[tuple] = None,
        feature_ndims: int = 1,
        active_dims: Optional[Union[int, Iterable]] = None,
        scale_diag: Optional[Union[ArrayLike, Number]] = None,
        **kwargs,
    ):
        self._length_scale_fn = length_scale_fn
        if fn_args is None:
            fn_args = tuple()
        self._fn_args = fn_args
        super(Gibbs, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs):
        return _Gibbs(
            length_scale_fn=self._length_scale_fn,
            fn_args=self._fn_args,
            feature_ndims=feature_ndims,
            **kwargs,
        )


@_build_docs
class Cosine(Covariance):
    r"""
    Cosine kernel.

    This kernel is part of the Periodic Kernels.
    It represents purely sinusoidal functions.

    .. math::

        k(x,x') = \sigma^2\cos\left(\frac{2\pi\|x-x'\|^2}{\ell^2}\right)

    where :math:`\sigma` is the ``amplitude`` and :math:`\ell` is the ``length_scale``.

    Parameters
    ----------
    %(_ls_amp_doc)
    %(_common_doc)

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import Cosine
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k = Cosine(1.)
    >>> k(x,x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.        , 0.47307032],
           [0.47307032, 1.        ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
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
        super(Cosine, self).__init__(
            feature_ndims=feature_ndims, active_dims=active_dims, scale_diag=scale_diag, **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return _Cosine(
            length_scale=self._length_scale,
            amplitude=self._amplitude,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def amplitude(self) -> Union[ArrayLike, float]:
        r"""Amplitude of the kernel function."""
        return self._amplitude


@_build_docs
class ScaledCov(Covariance):
    r"""
    Scaled Covariance Kernel.

    This kernel scales the covariance matrix given a scaling function and the kernel to scale.
    It can be used to create non-stationary kernels from stationary and periodic kernels.

    Parameters
    ----------
    kernel : pm.gp.cov.Covariance
        The covariance function to scale.
    scaling_fn : callable
        The scaling function.
    fn_args : tuple, optional
        Extra arguments to pass to the scaling function.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from pymc4.gp.cov import ExpQuad, ScaledCov
    >>> k = ExpQuad(1.)
    >>> fn = lambda x : tf.ones(x.shape)
    >>> k_scal = ScaledCov(k, fn)
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> k_scal(x, x)
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[2.        , 0.03663128],
           [0.03663128, 2.        ]], dtype=float32)>

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self, kernel: Covariance, scaling_fn: Callable, fn_args: Optional[tuple] = None, **kwargs,
    ):
        self._kernel = kernel
        self._scaling_fn = scaling_fn
        self._fn_args = fn_args
        super(ScaledCov, self).__init__(
            feature_ndims=kernel._feature_ndims,
            active_dims=kernel._active_dims,
            scale_diag=kernel._scale_diag,
            **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs) -> PositiveSemidefiniteKernel:
        return _ScaledCov(
            kernel=self._kernel._kernel,
            scaling_fn=self._scaling_fn,
            fn_args=self._fn_args,
            feature_ndims=feature_ndims,
            **kwargs,
        )

    @property
    def kernel(self):
        return self._kernel

    @property
    def scaling_fn(self):
        return self._scaling_fn

    @property
    def fn_args(self):
        return self._fn_args


class WarpedInput(Covariance):
    r"""
    Warped Input Kernel.

    Warp the inputs of any kernel using an arbitrary function defined
    using TensorFlow.

    .. math::
       k(x, x') = k(w(x), w(x'))

    Parameters
    ----------
    kernel : pm.gp.cov.Covariance
        The kernel function to warp
    warp_fn : callable
        TensorFlow function of ``X`` and additional optional arguments.
    fn_args : tuple, optional
        Additional inputs (besides X or Xs) to warp_func.

    Examples
    --------
    TODO

    Notes
    -----
    %(_note_doc)
    """

    def __init__(
        self, kernel: Covariance, warp_fn: Callable, fn_args: Optional[tuple] = None, **kwargs,
    ):
        self._kernel = kernel
        self._warp_fn = warp_fn
        if fn_args is None:
            fn_args = tuple()
        self._fn_args = fn_args
        super(WarpedInput, self).__init__(
            feature_ndims=kernel._feature_ndims,
            active_dims=kernel._active_dims,
            scale_diag=kernel._scale_diag,
            **kwargs,
        )

    def _init_kernel(self, feature_ndims: int, **kwargs):
        fn = lambda x, _, __: self._warp_fn(x, *self._fn_args)
        return FeatureTransformed(kernel=self._kernel._kernel, transformation_fn=fn, **kwargs)

    @property
    def kernel(self):
        return self._kernel

    @property
    def warp_fn(self):
        return self._warp_fn

    @property
    def fn_args(self):
        return self._fn_args
