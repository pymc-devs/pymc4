"""Execute graph for sMC with separate log probability collection for prior and likelihood.
Different logic for `proceed_distribution` method etc.
"""
from pymc4.flow.transformed_executor import TransformedSamplingExecutor


__all__ = ["SMCSamplingExecutor"]


class SMCSamplingExecutor(TransformedSamplingExecutor):
    """
        TODO: #229
    """

    def _modify_distribution_value_shapes_before_assert(self, value_shape, dist_shape):
        value_shape = value_shape[1:]
        dist_shape = dist_shape[1:]
        return value_shape, dist_shape
