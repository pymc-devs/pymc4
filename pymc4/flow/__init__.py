"""Functions for evaluating log probabilities."""
from .executor import SamplingExecutor, SamplingState
from .transformed_executor import TransformedSamplingExecutor
from .posterior_predictive_executor import PosteriorPredictiveSamplingExecutor
from .meta_executor import MetaSamplingExecutor

__all__ = [
    "SamplingExecutor",
    "TransformedSamplingExecutor",
    "PosteriorPredictiveSamplingExecutor",
    "MetaSamplingExecutor",
    "evaluate_model",
    "evaluate_model_transformed",
    "evaluate_model_posterior_predictive",
    "evaluate_meta_model",
]

evaluate_model = SamplingExecutor()
evaluate_model_transformed = TransformedSamplingExecutor()
evaluate_model_posterior_predictive = PosteriorPredictiveSamplingExecutor()
evaluate_meta_model = MetaSamplingExecutor()
