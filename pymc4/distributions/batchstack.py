"""The BatchStacker distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Union, Tuple

# Dependency imports
import numpy as np

# import tensorflow.compat.v2 as tf
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util


def _make_summary_statistic(attr):  # noqa
    """Build functions that compute summary statistics, eg, mean, stddev, mode."""

    def _fn(self, **kwargs):
        """Implement summary statistic, eg, mean, stddev, mode."""
        x = getattr(self.distribution, attr)(**kwargs)
        shape = prefer_static.concat(
            [
                prefer_static.ones(
                    prefer_static.rank_from_shape(self.batch_stack), dtype=self.batch_stack.dtype,
                ),
                self.distribution.batch_shape_tensor(),
                self.distribution.event_shape_tensor(),
            ],
            axis=0,
        )
        x = tf.reshape(x, shape=shape)
        shape = prefer_static.concat(
            [
                self.batch_stack,
                self.distribution.batch_shape_tensor(),
                self.distribution.event_shape_tensor(),
            ],
            axis=0,
        )
        return tf.broadcast_to(x, shape)

    return _fn


class BatchStacker(distribution_lib.Distribution):
    r"""
    BatchStacker distribution via independent draws.

    This distribution is useful for stacking collections of independent,
    identical draws. It is otherwise identical to the input distribution.

    The probability function is,

    .. math::
        p(x) = prod{ p(x[i]) : i = 0, ..., (n - 1) }

    Examples
    --------
    Example 1: Five scalar draws.

    >>> from tensorflow_probability import distributions as tfd
    >>> s = BatchStacker(
    ...     tfd.Normal(loc=0, scale=1),
    ...     batch_stack=5)
    >>> x = s.sample()
    >>> x.shape.as_list()
    [5]
    >>> lp = s.log_prob(x)
    >>> lp.shape.as_list()
    [5]

    Example 2: `[5, 4]`-draws of a bivariate Normal.

    >>> s = BatchStacker(
    ...     tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1),
    ...                     reinterpreted_batch_ndims=1),
    ...     batch_stack=[5, 4])
    >>> x = s.sample([6, 1])
    >>> x.shape.as_list()
    [6, 1, 5, 4, 3, 2]
    >>> lp = s.log_prob(x)
    >>> lp.shape.as_list()
    [6, 1, 5, 4, 3]
    """

    def __init__(
        self,
        distribution: tfd.Distribution,
        batch_stack=Union[int, Tuple[int, ...], tf.Tensor],
        validate_args: bool = False,
        name: Optional[str] = None,
    ):
        r"""
        Construct the `BatchStacker` distribution.

        Parameters
        ----------
        distribution : tfd.Distribution
            The base distribution instance to transform. Typically an instance of
            ``tensorflow_probability.distributions.Distribution``.
        batch_stack : Union[int, Tuple[int, ...], tf.Tensor]
            Shape of the stack of distributions. To be more precise, ``distribution`` has
            its underlying ``batch_shape``, what ``batch_stack`` does effectively is to
            add independent replications of ``distribution`` as extra ``batch_shape`` axes,
            to the left of the original ``batch_shape``. The resulting batch shape is
            ``tf.TensorShape(batch_stack) + distribution.batch_shape``.
        validate_args : bool
            Whether to validate input with asserts. If ``validate_args`` is ``False``,
            and the inputs are invalid, correct behavior is not guaranteed.
        name : Optional[str]
            The name for ops managed by the distribution. If ``None``, defaults to
            ``"BatchStacker" + distribution.name``.
        """
        parameters = dict(locals())
        name = name or "BatchStacker" + distribution.name
        self._distribution = distribution
        with tf.name_scope(name) as name:
            batch_stack = distribution_util.expand_to_vector(
                tf.convert_to_tensor(batch_stack, dtype_hint=tf.int32, name="batch_stack")
            )
            self._batch_stack = batch_stack
            super(BatchStacker, self).__init__(
                dtype=self._distribution.dtype,
                reparameterization_type=self._distribution.reparameterization_type,
                validate_args=validate_args,
                allow_nan_stats=self._distribution.allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    @property
    def distribution(self):
        """Get the underlying tensorflow_probability.Distribution."""
        return self._distribution

    @property
    def batch_stack(self):
        """Get this instance's batch_stack value."""
        return self._batch_stack

    def _batch_shape_tensor(self):
        return prefer_static.concat(
            [self.batch_stack, self.distribution.batch_shape_tensor(),], axis=0,
        )

    def _batch_shape(self):
        batch_stack = tf.TensorShape(tf.get_static_value(self.batch_stack))
        if (
            tensorshape_util.rank(batch_stack) is None
            or tensorshape_util.rank(self.distribution.event_shape) is None
        ):
            return tf.TensorShape(None)
        return tensorshape_util.concatenate(batch_stack, self.distribution.batch_shape)

    def _event_shape_tensor(self):
        return self.distribution.event_shape_tensor()

    def _event_shape(self):
        return self.distribution.event_shape

    def _sample_n(self, n, seed, **kwargs):
        return self.distribution.sample(
            prefer_static.concat([[n], self.batch_stack], axis=0), seed=seed, **kwargs
        )

    def _log_prob(self, x, **kwargs):
        batch_ndims = prefer_static.rank_from_shape(
            self.distribution.batch_shape_tensor, self.distribution.batch_shape
        )
        extra_batch_ndims = prefer_static.rank_from_shape(self.batch_stack)
        event_ndims = prefer_static.rank_from_shape(
            self.distribution.event_shape_tensor, self.distribution.event_shape
        )
        ndims = prefer_static.rank(x)
        # (1) Expand x's dims.
        d = ndims - extra_batch_ndims - batch_ndims - event_ndims
        x = tf.reshape(
            x,
            shape=tf.pad(
                tf.shape(x), paddings=[[prefer_static.maximum(0, -d), 0]], constant_values=1,
            ),
        )
        # (2) Compute x's log_prob.
        return self.distribution.log_prob(x, **kwargs)

    def _entropy(self, **kwargs):
        return self.distribution.entropy(**kwargs)

    _mean = _make_summary_statistic("mean")
    _stddev = _make_summary_statistic("stddev")
    _variance = _make_summary_statistic("variance")
    _mode = _make_summary_statistic("mode")


@kullback_leibler.RegisterKL(BatchStacker, BatchStacker)
def _kl_sample(a: BatchStacker, b: BatchStacker, name: str = "kl_sample") -> tf.Tensor:
    r"""
    Batched KL divergence :math:`KL(a || b)` for :class:`~.BatchStacker` distributions.

    We can leverage the fact that:

    .. math::
        KL(BatchStacker(a) || BatchStacker(b)) = \sum(KL(a || b))

    where the :math:`\sum` is over the ``batch_stack`` dims.

    Parameters
    ----------
    a : BatchStacker
        Instance of ``BatchStacker`` distribution.
    b : BatchStacker
        Instance of ``BatchStacker`` distribution.
    name : str
        Name to use for created ops.

    Returns
    -------
    kldiv : tf.Tensor
        Batchwise :math:`KL(a || b)`.

    Raises
    ------
    ValueError
        If the ``batch_stack`` of ``a`` and ``b`` don't match.
    """
    assertions = []
    a_ss = tf.get_static_value(a.batch_stack)
    b_ss = tf.get_static_value(b.batch_stack)
    msg = "`a.batch_stack` must be identical to `b.batch_stack`."
    if a_ss is not None and b_ss is not None:
        if not np.array_equal(a_ss, b_ss):
            raise ValueError(msg)
    elif a.validate_args or b.validate_args:
        assertions.append(assert_util.assert_equal(a.batch_stack, b.batch_stack, message=msg))
    with tf.control_dependencies(assertions):
        return kullback_leibler.kl_divergence(a.distribution, b.distribution, name=name)
