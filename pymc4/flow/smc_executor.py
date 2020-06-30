"""Execute graph for sMC with separate log probability collection for prior and likelihood.
Different logic for `proceed_distribution` method etc.
"""
from typing import Mapping, Any
from pymc4.distributions import distribution
from pymc4.flow.transformed_executor import TransformedSamplingExecutor, transform_dist_if_necessary
from pymc4.flow.executor import ModelType, SamplingState


__all__ = ["SMCSamplingExecutor"]


class SMCSamplingExecutor(TransformedSamplingExecutor):
    """
        TODO: #229
    """

    def modify_distribution(
        self, dist: ModelType, model_info: Mapping[str, Any], state: SamplingState
    ) -> ModelType:
        """Apply transformations to a distribution."""
        dist = super().modify_distribution(dist, model_info, state)
        if not isinstance(dist, distribution.Distribution):
            return dist

        return transform_dist_if_necessary(
            dist, state, allow_transformed_and_untransformed=True, is_smc=True
        )

    def _sample_unobserved(self, dist, state, scoped_name, sample_func, num_chains, sample_shape):
        # We have smc_run flag for the sMC graph evaluation
        # For sMC run we need to store batch values in the state object
        # to avoid singularity of prior samples when tiling (mcmc)
        if dist.is_root:
            sample_ = sample_func(sample_shape=(num_chains,) + sample_shape)
            state.untransformed_values_batched[scoped_name] = sample_
            return_value = state.untransformed_values[scoped_name] = sample_func(sample_shape)
        else:
            sample_ = sample_func(sample_shape=(num_chains,))
            state.untransformed_values_batched[scoped_name] = sample_
            return_value = state.untransformed_values[scoped_name] = sample_func()
        return state, return_value
