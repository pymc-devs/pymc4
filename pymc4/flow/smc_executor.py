"""
    Execute graph for sMC with separate log probability collection for
    prior and likelihood. Different logic for `proceed_distribution` method
    etc.
"""
import functools
import tensorflow as tf
from typing import Any, Callable, Tuple, Union
from pymc4.flow.transformed_executor import TransformedSamplingExecutor, transform_dist_if_necessary
from pymc4.flow.executor import ModelType, SamplingState


__all__ = ["SMCSamplingExecutor"]


class SMCSamplingExecutor(TransformedSamplingExecutor):
    """
    Subclass of `TransformedSamplingExecutor` which is used for
    inference in sMC. Adds custom method to sample unobserved
    variables in the first execution of the model.
    """

    def __init__(self):
        super().__init__()
        self.transform_dist_if_necessary = functools.partial(
            transform_dist_if_necessary, is_smc=True
        )

    def _sample_unobserved(
        self,
        dist: ModelType,
        state: SamplingState,
        scoped_name: str,
        sample_func: Callable,
        *,
        sample_shape: Union[int, Tuple[int], tf.TensorShape] = None,
        draws: int = None,
    ) -> Tuple[SamplingState, Any]:
        # For sMC run we need to store batch values in the state object
        # to avoid singularity of prior samples when tiling (mcmc)
        if dist.is_root:
            sample_ = sample_func(sample_shape=(draws,) + sample_shape)
            state.untransformed_values_batched[scoped_name] = sample_
            return_value = state.untransformed_values[scoped_name] = sample_func(sample_shape)
        else:
            sample_ = sample_func(sample_shape=(draws,))
            state.untransformed_values_batched[scoped_name] = sample_
            return_value = state.untransformed_values[scoped_name] = sample_func()
        return state, return_value
