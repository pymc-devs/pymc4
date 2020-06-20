"""Execute graph with test values to extract a model's meta-information.

Specifically, we wish to extract:

- All variable's core shapes
- All observed, deterministic, and unobserved variables (both transformed and
untransformed.
"""
from pymc4.flow.transformed_executor import TransformedSamplingExecutor
from pymc4.flow.posterior_predictive_executor import PosteriorPredictiveSamplingExecutor


__all__ = ["MetaSamplingExecutor", "MetaPosteriorPredictiveSamplingExecutor"]


class MetaSamplingExecutor(TransformedSamplingExecutor):
    """
        Do a forward pass through the model only using distribution test values.
    """

    def __init__(self):
        self.mode = "meta"


class MetaPosteriorPredictiveSamplingExecutor(
    MetaSamplingExecutor, PosteriorPredictiveSamplingExecutor
):
    """
        Do a forward pass through the model only using distribution test values.
        Also modify the distributions to make them suitable for posterior predictive sampling.
    """

    # Everything is done in the parent classes
    pass
