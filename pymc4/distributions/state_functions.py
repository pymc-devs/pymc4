import abc
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

tfd = tfp.distributions

__all__ = ["categorical_uniform_fn", "bernoulli_fn", "gaussian_round_fn"]

# TODO: We can furthere optimize proposal function
# For now if a user provides two sampler functions with
# the same proposal function but different scale parameter
# the sampler code will treat them as separete sampling kernels
# which will increase the graph size.


def wrap_inner_fn_name():
    """
        Change the name of nested function.
        We need this to compare the proposal functions.
    """

    def decorate(func):
        def call(*args, **kwargs):
            q = func(*args, **kwargs)
            q.__name__ = "/".join([func.__name__, q.__name__])
            return q

        return call

    return decorate


class Proposal(metaclass=abc.ABCMeta):
    def __init__(self, name=None):
        if name:
            self._name = name

    @abc.abstractmethod
    def _fn(self, state_parts, seed):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def __call__(self):
        return self._fn


class CategoricalUniformFn(Proposal):
    """Returns a callable that samples new proposal from Categorical distribution with uniform probabilites
    Args:
       classes: tuple, tf.TensorShape
           Shape of logits/probs parameter of the distribution
       name: Python `str` name prefixed to Ops created by this function.
           Default value: 'categorical_uniform_fn'.
    Returns:
        categorical_uniform_fn: A callable accepting a Python `list` of `Tensor`s
         representing the state parts of the `current_state` and an `int`
         representing the random seed used to generate the proposal. The callable
         returns the same-type `list` of `Tensor`s as the input and represents the
         proposal for the RWM algorithm.
    """

    _name = "categorical_uniform_fn"

    def __init__(self, classes, name=None):
        super().__init__(name)
        self.classes = classes

    def _fn(self, state_parts, seed):
        with tf.name_scope(self._name or "categorical_uniform_fn"):
            deltas = tf.unstack(
                tf.map_fn(
                    lambda x: tfd.Categorical(logits=tf.ones(self.classes)).sample(
                        seed=seed, sample_shape=tf.shape(x)
                    ),
                    tf.stack(state_parts),
                )
            )
            return deltas

    def __eq__(self, other):
        return self._name == other._name and self.classes == other.classes


class BernoulliFn(Proposal):
    """Returns a callable that samples new proposal from Bernoulli distribution
    Args:
       scale: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
         controlling the upper and lower bound of the uniform proposal
         distribution.
       name: Python `str` name prefixed to Ops created by this function.
           Default value: 'bernoulli_fn'.
    Returns:
        bernoulli_fn: A callable accepting a Python `list` of `Tensor`s
         representing the state parts of the `current_state` and an `int`
         representing the random seed used to generate the proposal. The callable
         returns the same-type `list` of `Tensor`s as the input and represents the
         proposal for the RWM algorithm.
    """

    _name = "bernoulli_fn"

    def __init__(self, scale=1.0, name=None):
        super().__init__(name)
        self.scale = scale

    def _fn(self, state_parts, seed):
        scale = self.scale
        with tf.name_scope(self._name or "bernoulli_fn"):
            scales = scale if mcmc_util.is_list_like(scale) else [scale]
            if len(scales) == 1:
                scales *= len(state_parts)
            if len(state_parts) != len(scales):
                raise ValueError("`scale` must broadcast with `state_parts`")

            def generate_new_values(state_part, scale_part):
                # TODO: is there a more elegant way
                ndim = scale_part.get_shape().ndims
                reduced_elem = tf.squeeze(
                    tf.slice(
                        scale_part, tf.zeros(ndim, dtype=tf.int32), tf.ones(ndim, dtype=tf.int32),
                    )
                )
                delta = tfd.Bernoulli(probs=0.5 * reduced_elem, dtype=state_part.dtype).sample(
                    seed=seed, sample_shape=(tf.shape(state_part))
                )
                state_part += delta
                state_part = state_part % 2.0
                return state_part

            state_parts = tf.stack(state_parts)
            orig_dtype = state_parts.dtype
            # TODO: we create scale_part with shape=state_part.shape
            # each function call. But scalar value would be enough

            shape_ = state_parts.shape
            inds_tile = tf.concat(
                [tf.constant([shape_[0]]), tf.ones(shape_.ndims, dtype=tf.int32)], axis=0
            )
            shape_ = (1,) + shape_[1:] + (shape_[0],)
            scales = tf.tile(tf.broadcast_to(scales, shape_), inds_tile)[..., 0]
            state_parts = tf.cast(state_parts, dtype=scales.dtype)
            deltas = tf.unstack(
                tf.map_fn(
                    lambda x: generate_new_values(
                        x[0], x[1]
                    ),  # TODO: some issues with unpack and tf graph
                    tf.stack([state_parts, scales], axis=1),
                )
            )
            return tf.unstack(tf.cast(deltas, dtype=orig_dtype))

    def __eq__(self, other):
        return self._name == other._name and self.scale == other.scale


class GaussianRoundFn(Proposal):
    """Returns a callable that samples new proposal from Normal distribution with round
    Args:
       scale: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
         controlling the upper and lower bound of the uniform proposal
         distribution.
       name: Python `str` name prefixed to Ops created by this function.
           Default value: 'gaussian_round_fn'.
    Returns:
        gaussian_round_fn: A callable accepting a Python `list` of `Tensor`s
         representing the state parts of the `current_state` and an `int`
         representing the random seed used to generate the proposal. The callable
         returns the same-type `list` of `Tensor`s as the input and represents the
         proposal for the RWM algorithm.
    """

    _name = "gaussian_round_fn"

    def __init__(self, scale=1.0, name=None):
        super().__init__(name)
        self.scale = scale

    def _fn(self, state_parts, seed):
        scale = self.scale
        with tf.name_scope(self._name or "bernoulli_uniform_fn"):
            scales = scale if mcmc_util.is_list_like(scale) else [scale]
            if len(scales) == 1:
                scales *= len(state_parts)
            if len(state_parts) != len(scales):
                raise ValueError("`scale` must broadcast with `state_parts`")

            def generate_new_values(state_part, scale_part):
                ndim = scale_part.get_shape().ndims
                reduced_elem = tf.squeeze(
                    tf.slice(
                        scale_part, tf.zeros(ndim, dtype=tf.int32), tf.ones(ndim, dtype=tf.int32),
                    )
                )
                delta = tfd.Normal(0.0, reduced_elem * 1.0).sample(
                    seed=seed, sample_shape=(tf.shape(state_part))
                )
                state_part += delta
                return tf.round(state_part)

            state_parts = tf.stack(state_parts)

            shape_ = state_parts.shape
            inds_tile = tf.concat(
                [tf.constant([shape_[0]]), tf.ones(shape_.ndims, dtype=tf.int32)], axis=0
            )
            shape_ = (1,) + shape_[1:] + (shape_[0],)
            scales = tf.tile(tf.broadcast_to(scales, shape_), inds_tile)[..., 0]

            deltas = tf.unstack(
                tf.map_fn(
                    lambda x: generate_new_values(x[0], x[1]),
                    tf.stack([state_parts, scales], axis=1),
                )
            )
            return tf.unstack(deltas)

    def __eq__(self, other):
        return self._name == other._name and self.scale == other.scale


categorical_uniform_fn = CategoricalUniformFn
bernoulli_fn = BernoulliFn
gaussian_round_fn = GaussianRoundFn
