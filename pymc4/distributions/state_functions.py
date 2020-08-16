import abc
from typing import Optional, List, Union, Any
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


class Proposal(metaclass=abc.ABCMeta):
    def __init__(self, name: Optional[str] = None):
        if name:
            self._name = name

    @abc.abstractmethod
    def _fn(self, state_parts: List[tf.Tensor], seed: Optional[int]) -> List[tf.Tensor]:
        """
        Proposal function that is passed as the argument
        to RWM kernel

        Parameters
        ----------
        state_parts : List[tf.Tensor]
            A list of `Tensor`s of any shape and real dtype representing
            the state of the `current_state` of the Markov chain
        seed: Optional[int]
            The random seed for this `Op`. If `None`, no seed is
        applied
            Default value: `None`

        Returns
        -------
        List[tf.Tensor]
            A Python `list` of The `Tensor`s. Has the same
            shape and type as the `state_parts`.

        Raises
        ------
        ValueError: if `scale` does not broadcast with `state_parts`.
        """
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """
        Comparison operator overload of each proposal sub-class.
        The operator is required to disnguish same proposal functions to separate
        samplers in `Compound step`

        Parameters
        ----------
        other: pm.distributions.Proposal
        Another instance of `Proposal` sub-class.

        Returns
        -------
        bool
            True/False for equality of instances
        """
        pass

    def __call__(self):
        return self._fn


class CategoricalUniformFn(Proposal):
    """
    Categorical proposal sub-class with the `_fn` that is sampling new proposal
    from catecorical distribution with uniform probabilities.

    Parameters
    ----------
    classes: int
        Number of classes for catecorical distribution
    name: Optional[str]
        Python `str` name prefixed to Ops created by this function.
        Default value: 'categorical_uniform_fn'.
    """

    _name = "categorical_uniform_fn"

    def __init__(self, classes: int, name: Optional[str] = None):
        super().__init__(name)
        self.classes = classes

    def _fn(self, state_parts: List[tf.Tensor], seed: Optional[int]) -> List[tf.Tensor]:
        with tf.name_scope(self._name or "categorical_uniform_fn"):
            deltas = tf.nest.map_structure(
                lambda x: tfd.Categorical(logits=tf.ones(self.classes)).sample(
                    seed=seed, sample_shape=tf.shape(x)
                ),
                state_parts,
            )
            return deltas

    def __eq__(self, other) -> bool:
        return self._name == other._name and self.classes == other.classes


class BernoulliFn(Proposal):
    """
    Bernoulli proposal sub-class with the `_fn` that is sampling new proposal
    from bernoulli distribution with p=0.5.

    Parameters
    ----------
    name: Optional[str]
        Python `str` name prefixed to Ops created by this function.
        Default value: 'categorical_uniform_fn'.
    """

    _name = "bernoulli_fn"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def _fn(self, state_parts: List[tf.Tensor], seed: Optional[int]) -> List[tf.Tensor]:
        with tf.name_scope(self._name or "bernoulli_fn"):

            def generate_bernoulli(state_part):
                delta = tfd.Bernoulli(
                    probs=tf.ones_like(state_part, dtype=tf.float32) * 0.5, dtype=state_part.dtype
                ).sample(seed=seed)
                state_part = (state_part + delta) % tf.constant(2, dtype=state_part.dtype)
                return state_part

            new_state = tf.nest.map_structure(generate_bernoulli, state_parts)
            return new_state

    def __eq__(self, other) -> bool:
        return self._name == other._name


class GaussianRoundFn(Proposal):
    """
    Gaussian-Round proposal sub-class with the `_fn` that is sampling new proposal
    from normal distribution N(0, 1) and rounding the values.

    Parameters
    ----------
    scale: Union[List[Any], Any]
        a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
        controlling the scale of the proposal distribution.
    name: Optional[str]
        Python `str` name prefixed to Ops created by this function.
        Default value: 'categorical_uniform_fn'.
    """

    _name = "gaussian_round_fn"

    def __init__(self, scale: Union[List[Any], Any] = 1.0, name: Optional[str] = None):
        super().__init__(name)
        self.scale = scale

    def _fn(self, state_parts: List[tf.Tensor], seed: Optional[int]) -> List[tf.Tensor]:
        scale = self.scale
        with tf.name_scope(self._name or "gaussian_round_fn"):
            scales = scale if mcmc_util.is_list_like(scale) else [scale]
            if len(scales) == 1:
                scales *= len(state_parts)
            if len(state_parts) != len(scales):
                raise ValueError("`scale` must broadcast with `state_parts`")

            def generate_rounded_normal(state_part, scale_part):
                delta = tfd.Normal(0.0, tf.ones_like(state_part)).sample(seed=seed)
                state_part += delta
                return tf.round(state_part)

            new_state = tf.nest.map_structure(generate_rounded_normal, state_parts, scales)
            return new_state

    def __eq__(self, other) -> bool:
        return self._name == other._name and self.scale == other.scale


categorical_uniform_fn = CategoricalUniformFn
bernoulli_fn = BernoulliFn
gaussian_round_fn = GaussianRoundFn
