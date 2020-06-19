"""Half-Student's t distribution class."""

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.util.seed_stream import SeedStream

__all__ = ["HalfStudentT"]


class HalfStudentT(distribution.Distribution):
    r"""
    Half-Student's t distribution.

    The half-Student's t distribution is parameterized by degree of freedom ``df``,
    location ``loc``, and ``scale``. It represents the right half of the two symmetric
    halves in a [Student's t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution).

    The probability density function (pdf) for the half-Student's t distribution
    is given by

    .. math::

        pdf(x \mid \nu, \mu, \sigma) = 2 \frac{(\frac{1 + y^2}{\nu})^{-0.5 (\nu + 1)}}{Z}

    where
    :math:`y = \frac{x - \mu}{\sigma}`,
    :math:`Z = \|\sigma\| \sqrt(\nu \pi) \Gamma(0.5 \nu) / \Gamma(0.5 (\nu + 1))`

    where :math:`\Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

    The support of the distribution is given by the interval :math:`[loc, infinity)`.
    """

    def __init__(
        self, df, loc, scale, validate_args=False, allow_nan_stats=True, name="HalfStudentT"
    ):
        r"""
        Construct a half-Student's t distribution with ``df``, ``loc`` and ``scale``.

        Parameters
        ----------
        df: Floating-point ``Tensor``
            The degrees of freedom of the distribution(s).
            ``df`` must contain only positive values.
        loc: Floating-point ``Tensor``
            The location(s) of the distribution(s).
        scale: Floating-point ``Tensor``
            The scale(s) of the distribution(s).
            Must contain only positive values.
        validate_args: bool, optional
            When ``True`` distribution parameters are checked for validity despite
            possibly degrading runtime performance. When ``False`` invalid inputs
            may silently render incorrect outputs.
            Default value: ``False`` (i.e. do not validate args).
        allow_nan_stats: bool, optional
            When ``True``, statistics (e.g., mean, mode, variance) use the value
            "``NaN``" to indicate the result is undefined. When ``False``, an
            exception is raised if one or more of the statistic's batch members
            are undefined. Default value: ``True``.
        name: str, optional
            Name prefixed to Ops created by this class.
            Default value: 'HalfStudentT'.

        Raises
        ------
            TypeError: if ``df``, ``loc``, or ``scale`` are different dtypes
        """

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([df, loc, scale], dtype_hint=tf.float32)
            self._df = tensor_util.convert_nonref_to_tensor(df, name="df", dtype=dtype)
            self._loc = tensor_util.convert_nonref_to_tensor(loc, name="loc", dtype=dtype)
            self._scale = tensor_util.convert_nonref_to_tensor(scale, name="scale", dtype=dtype)
            dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
            super(HalfStudentT, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(("df", "loc", "scale"), ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3))
        )

    @classmethod
    def _params_event_ndims(cls):
        return dict(df=0, loc=0, scale=0)

    @property
    def df(self):
        """Degrees of freedom parameters."""
        return self._df

    @property
    def loc(self):
        """Distribution parameter for the location."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the scale."""
        return self._scale

    def _batch_shape_tensor(self, df=None, loc=None, scale=None):
        return prefer_static.broadcast_shape(
            prefer_static.shape(self.df if df is None else df),
            prefer_static.broadcast_shape(
                prefer_static.shape(self.loc if loc is None else loc),
                prefer_static.shape(self.scale if scale is None else scale),
            ),
        )

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            tf.broadcast_static_shape(self.df.shape, self.loc.shape), self.scale.shape
        )

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n, seed=None):
        # The sampling method comes from the fact that if:
        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = |X| / sqrt(Z / df)
        # then:
        #   Y ~ HalfStudentT(df).
        df = tf.convert_to_tensor(self.df)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        batch_shape = self._batch_shape_tensor(df=df, loc=loc, scale=scale)
        shape = tf.concat([[n], batch_shape], 0)
        seed = SeedStream(seed, "half_student_t")

        abs_normal_sample = tf.math.abs(tf.random.normal(shape, dtype=self.dtype, seed=seed()))
        df = df * tf.ones(batch_shape, dtype=self.dtype)
        gamma_sample = tf.random.gamma([n], 0.5 * df, beta=0.5, dtype=self.dtype, seed=seed())
        samples = abs_normal_sample * tf.math.rsqrt(gamma_sample / df)
        return samples * scale + loc  # Abs(scale) not wanted.

    def _log_prob(self, x):
        df = tf.convert_to_tensor(self.df)
        scale = tf.convert_to_tensor(self.scale)
        loc = tf.convert_to_tensor(self.loc)
        y = (x - loc) / scale  # Abs(scale) superfluous.
        log_unnormalized_prob = -0.5 * (df + 1.0) * tf.math.log1p(y ** 2.0 / df)
        log_normalization = (
            tf.math.log(tf.abs(scale))
            + 0.5 * tf.math.log(df)
            + 0.5 * np.log(np.pi)
            + tf.math.lgamma(0.5 * df)
            - tf.math.lgamma(0.5 * (df + 1.0))
            - np.log(2.0)
        )
        log_prob = log_unnormalized_prob - log_normalization
        return tf.where(x < loc, dtype_util.as_numpy_dtype(self.dtype)(-np.inf), log_prob)

    def _cdf(self, x):
        df = tf.convert_to_tensor(self.df)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        safe_x = self._get_safe_input(x, loc=loc, scale=scale)
        # Take Abs(scale) to make subsequent where work correctly.
        y = (safe_x - loc) / tf.abs(scale)
        x_t = df / (y ** 2.0 + df)
        neg_cdf = 0.5 * tf.math.betainc(
            0.5 * tf.broadcast_to(df, prefer_static.shape(x_t)), 0.5, x_t
        )
        return tf.where(x < loc, dtype_util.as_numpy_dtype(self.dtype)(-np.inf), 2.0 - 2 * neg_cdf)

    @distribution_util.AppendDocstring(
        r"""
        The mean of half Student's T equals

        .. math::

            2 \\sigma \\frac{\\sqrt(\\frac{\\nu}{pi})\\Gamma(\\frac{df + 1}{2})}{\\(Gamma(df/2 (df - 1))}

        if ``df > 1``, otherwise it is ``NaN``. If ``self.allow_nan_stats=True``, then
        an exception will be raised rather than returning ``NaN``."""
    )
    def _mean(self):
        df = tf.convert_to_tensor(self.df)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        log_mean = (
            np.log(2)
            + tf.math.log(scale)
            + 0.5 * (tf.math.log(df) - np.log(np.pi))
            + tf.math.lgamma(0.5 * (df + 1.0))
            - tf.math.lgamma(0.5 * df)
            - tf.math.log(df - 1)
        )
        mean = tf.math.exp(log_mean)
        if self.allow_nan_stats:
            return tf.where(df > 1.0, mean, dtype_util.as_numpy_dtype(self.dtype)(np.nan))
        else:
            return distribution_util.with_dependencies(
                [
                    assert_util.assert_less(
                        tf.ones([], dtype=self.dtype),
                        df,
                        message="mean not defined for components of df <= 1",
                    )
                ],
                mean,
            )

    @distribution_util.AppendDocstring(
        """
        The variance for half Student's T equals

        ``scale^2 (df / (df - 2) - 4 df / (pi (df - 1)^2)(Gamma((df + 1) / 2)) / Gamma(df / 2))^2)``

        when ``df > 2``
        ``infinity``, when ``1 < df <= 2``
        ``NaN``, when ``df <= 1``
        """
    )
    def _variance(self):
        df = tf.convert_to_tensor(self.df)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        log_correction = (
            np.log(4)
            + tf.math.log(df)
            - np.log(np.pi)
            - 2.0 * tf.math.log(df - 1)
            + 2.0 * tf.math.lgamma(0.5 * (df + 1))
            - 2.0 * tf.math.lgamma(0.5 * df)
        )
        var = tf.math.square(scale) * (df / (df - 2.0) - tf.math.exp(log_correction))
        result_where_defined = tf.where(
            df > 2.0, var, dtype_util.as_numpy_dtype(self.dtype)(np.inf)
        )
        if self.allow_nan_stats:
            return tf.where(
                df > 1.0, result_where_defined, dtype_util.as_numpy_dtype(self.dtype)(np.nan)
            )
        else:
            return distribution_util.with_dependencies(
                [
                    assert_util.assert_less(
                        tf.ones([], dtype=self.dtype),
                        df,
                        message="variance not defined for components of df <= 1",
                    )
                ],
                result_where_defined,
            )

    def _get_safe_input(self, x, loc, scale):
        safe_value = 0.5 * scale + loc
        return tf.where(x < loc, safe_value, x)

    def _sample_control_dependencies(self, x):
        """Check the validity of a sample."""
        assertions = []
        if not self.validate_args:
            return assertions
        loc = tf.convert_to_tensor(self.loc)
        assertions.append(
            assert_util.assert_greater_equal(
                x, loc, message="Sample must be greater than or equal to `loc`."
            )
        )
        return assertions

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        if is_init != tensor_util.is_ref(self.df):
            assertions.append(
                assert_util.assert_positive(self.df, message="Argument `df` must be positive.")
            )
        if is_init != tensor_util.is_ref(self.scale):
            assertions.append(
                assert_util.assert_positive(
                    self.scale, message="Argument `scale` must be positive."
                )
            )
        return assertions
