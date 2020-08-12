import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

tfd = tfp.distributions

__all__ = ["categorical_uniform_fn", "bernoulli_fn", "gaussian_round_fn", "poisson_fn"]


def categorical_uniform_fn(event_shape, name=None):
    """Returns a callable that samples new proposal from Categorical distribution with uniform probabilites
    Args:
       event_shape: tuple, tf.TensorShape
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

    def _fn(state_parts, seed):
        with tf.name_scope(name or "categorical_uniform_fn"):
            deltas = tf.unstack(
                tf.map_fn(
                    lambda x: tfd.Categorical(logits=tf.ones(event_shape)).sample(
                        seed=seed, sample_shape=tf.shape(x)
                    ),
                    tf.stack(state_parts),
                )
            )
            return deltas

    return _fn


def bernoulli_fn(scale=1.0, name=None):
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

    def _fn(state_parts, seed):
        with tf.name_scope(name or "bernoulli_fn"):
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
                        scale_part,
                        tf.zeros(ndim, dtype=tf.int32),
                        tf.ones(ndim, dtype=tf.int32),
                    )
                )
                delta = tfd.Bernoulli(
                    probs=0.5 * reduced_elem, dtype=state_part.dtype
                ).sample(seed=seed, sample_shape=(tf.shape(state_part)))
                state_part += delta
                state_part = state_part % 2.0
                return state_part

            state_parts = tf.stack(state_parts)
            orig_dtype = state_parts.dtype
            # TODO: we create scale_part with shape=state_part.shape
            # each function call. But scalar value would be enough
            scales = tf.broadcast_to(scales, state_parts.shape)
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

    return _fn


def gaussian_round_fn(scale=1.0, name=None):
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

    def _fn(state_parts, seed):
        with tf.name_scope(name or "bernoulli_uniform_fn"):
            scales = scale if mcmc_util.is_list_like(scale) else [scale]
            if len(scales) == 1:
                scales *= len(state_parts)
            if len(state_parts) != len(scales):
                raise ValueError("`scale` must broadcast with `state_parts`")

            def generate_new_values(state_part, scale_part):
                ndim = scale_part.get_shape().ndims
                reduced_elem = tf.squeeze(
                    tf.slice(
                        scale_part,
                        tf.zeros(ndim, dtype=tf.int32),
                        tf.ones(ndim, dtype=tf.int32),
                    )
                )
                delta = tfd.Normal(0.0, reduced_elem * 1.0).sample(
                    seed=seed, sample_shape=(tf.shape(state_part))
                )
                state_part += delta
                return tf.round(state_part)

            state_parts = tf.stack(state_parts)
            # TODO: we create scale_part with shape=state_part.shape
            # each function call. But scalar value would be enough
            scales = tf.broadcast_to(scales, state_parts.shape)
            deltas = tf.unstack(
                tf.map_fn(
                    lambda x: generate_new_values(x[0], x[1]),
                    tf.stack([state_parts, scales], axis=1),
                )
            )
            return tf.unstack(deltas)

    return _fn
