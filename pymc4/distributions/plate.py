"""The Plate distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

# import tensorflow.compat.v2 as tf
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util


def _make_summary_statistic(attr):
    """Factory for implementing summary statistics, eg, mean, stddev, mode."""

    def _fn(self, **kwargs):
        """Implements summary statistic, eg, mean, stddev, mode."""
        x = getattr(self.distribution, attr)(**kwargs)
        shape = prefer_static.concat(
            [
                prefer_static.ones(
                    prefer_static.rank_from_shape(self.plate_shape), dtype=self.plate_shape.dtype
                ),
                self.distribution.batch_shape_tensor(),
                self.distribution.event_shape_tensor(),
            ],
            axis=0,
        )
        x = tf.reshape(x, shape=shape)
        shape = prefer_static.concat(
            [
                self.plate_shape,
                self.distribution.batch_shape_tensor(),
                self.distribution.event_shape_tensor(),
            ],
            axis=0,
        )
        return tf.broadcast_to(x, shape)

    return _fn


class Plate(distribution_lib.Distribution):
    """Plate distribution via independent draws.
  This distribution is useful for stacking collections of independent,
  identical draws. It is otherwise identical to the input distribution.
  #### Mathematical Details
  The probability function is,
  ```none
  p(x) = prod{ p(x[i]) : i = 0, ..., (n - 1) }
  ```
  #### Examples
  ```python
  tfd = tfp.distributions
  # Example 1: Five scalar draws.
  s = tfd.Plate(
      tfd.Normal(loc=0, scale=1),
      plate_shape=5)
  x = s.sample()
  # ==> x.shape: [5]
  lp = s.log_prob(x)
  # ==> lp.shape: [5]
  #
  # Example 2: `[5, 4]`-draws of a bivariate Normal.
  s = tfd.Plate(
      tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1),
                      reinterpreted_batch_ndims=1),
      plate_shape=[5, 4])
  x = s.sample([6, 1])
  # ==> x.shape: [6, 1, 5, 4, 3, 2]
  lp = s.log_prob(x)
  # ==> lp.shape: [6, 1, 5, 4, 3]
  ```
  """

    def __init__(self, distribution, plate_shape=(), validate_args=False, name=None):
        """Construct the `Plate` distribution.
    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      plate_shape: `int` scalar or vector `Tensor` representing the shape of a
        single sample.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Plate' + distribution.name`).
    """
        parameters = dict(locals())
        name = name or "Plate" + distribution.name
        self._distribution = distribution
        with tf.name_scope(name) as name:
            plate_shape = distribution_util.expand_to_vector(
                tf.convert_to_tensor(plate_shape, dtype_hint=tf.int32, name="plate_shape")
            )
            self._plate_shape = plate_shape
            super(Plate, self).__init__(
                dtype=self._distribution.dtype,
                reparameterization_type=self._distribution.reparameterization_type,
                validate_args=validate_args,
                allow_nan_stats=self._distribution.allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    @property
    def distribution(self):
        return self._distribution

    @property
    def plate_shape(self):
        return self._plate_shape

    def _batch_shape_tensor(self):
        return prefer_static.concat(
            [self.plate_shape, self.distribution.batch_shape_tensor(),], axis=0
        )

    def _batch_shape(self):
        plate_shape = tf.TensorShape(tf.get_static_value(self.plate_shape))
        if (
            tensorshape_util.rank(plate_shape) is None
            or tensorshape_util.rank(self.distribution.event_shape) is None
        ):
            return tf.TensorShape(None)
        return tensorshape_util.concatenate(plate_shape, self.distribution.batch_shape)

    def _event_shape_tensor(self):
        return self.distribution.event_shape_tensor()

    def _event_shape(self):
        return self.distribution.event_shape

    def _sample_n(self, n, seed, **kwargs):
        return self.distribution.sample(
            prefer_static.concat([[n], self.plate_shape], axis=0), seed=seed, **kwargs
        )

    def _log_prob(self, x, **kwargs):
        batch_ndims = prefer_static.rank_from_shape(
            self.distribution.batch_shape_tensor, self.distribution.batch_shape
        )
        extra_plate_ndims = prefer_static.rank_from_shape(self.plate_shape)
        event_ndims = prefer_static.rank_from_shape(
            self.distribution.event_shape_tensor, self.distribution.event_shape
        )
        ndims = prefer_static.rank(x)
        # (1) Expand x's dims.
        d = ndims - extra_plate_ndims - batch_ndims - event_ndims
        x = tf.reshape(
            x,
            shape=tf.pad(
                tf.shape(x), paddings=[[prefer_static.maximum(0, -d), 0]], constant_values=1
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


@kullback_leibler.RegisterKL(Plate, Plate)
def _kl_sample(a, b, name="kl_sample"):
    """Batched KL divergence `KL(a || b)` for Plate distributions.
  We can leverage the fact that:
  ```
  KL(Plate(a) || Plate(b)) = sum(KL(a || b))
  ```
  where the sum is over the `plate_shape` dims.
  Args:
    a: Instance of `Plate` distribution.
    b: Instance of `Plate` distribution.
    name: (optional) name to use for created ops.
      Default value: `"kl_sample"`'.
  Returns:
    kldiv: Batchwise `KL(a || b)`.
  Raises:
    ValueError: If the `plate_shape` of `a` and `b` don't match.
  """
    assertions = []
    a_ss = tf.get_static_value(a.plate_shape)
    b_ss = tf.get_static_value(b.plate_shape)
    msg = "`a.plate_shape` must be identical to `b.plate_shape`."
    if a_ss is not None and b_ss is not None:
        if not np.array_equal(a_ss, b_ss):
            raise ValueError(msg)
    elif a.validate_args or b.validate_args:
        assertions.append(assert_util.assert_equal(a.plate_shape, b.plate_shape, message=msg))
    with tf.control_dependencies(assertions):
        return kullback_leibler.kl_divergence(a.distribution, b.distribution, name=name)
