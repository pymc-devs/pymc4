"""Execute graph in a transformed space and change the observed distribution's shape.

Specifically, we wish to transform the observed distributions' shape to make
it aware of the observed values, in order to later draw posterior predictive
samples.
"""
from typing import Mapping, Any
import tensorflow as tf
from pymc4 import scopes
from pymc4.distributions.distribution import Distribution
from pymc4.flow.executor import (
    ModelType,
    SamplingState,
    observed_value_in_evaluation,
    get_observed_tensor_shape,
    assert_observations_compatible_with_distribution_shape,
)
from pymc4.flow.transformed_executor import TransformedSamplingExecutor


class PosteriorPredictiveSamplingExecutor(TransformedSamplingExecutor):
    """Execute the probabilistic model for posterior predictive sampling.

    This means that the model will be evaluated in the same way as the
    TransformedSamplingExecutor evaluates it. All unobserved distributions
    will be left as they are. All observed distributions will modified in
    the following way:
    1) The distribution's shape (batch_shape + event_shape) will be checked
    for consitency with the supplied observed value's shape.
    2) If they are inconsistent, an EvaluationError will be raised.
    3) If they are consistent the distribution's observed values shape
    will be broadcasted with the distribution's shape to construct a new
    Distribution instance with no observations. This distribution will be
    used for posterior predictive sampling
    """

    def modify_distribution(
        self, dist: ModelType, model_info: Mapping[str, Any], state: SamplingState
    ) -> ModelType:
        """Remove the observed distribution values but keep their shapes.

        Modify observed Distribution instances in the following way:
        1) The distribution's shape (batch_shape + event_shape) will be checked
        for consitency with the supplied observed value's shape.
        2) If they are inconsistent, an EvaluationError will be raised.
        3) If they are consistent the distribution's observed values' shape
        will be broadcasted with the distribution's shape to construct a new
        Distribution instance with no observations.
        4) This distribution will be yielded instead of the original incoming
        dist, and it will be used for posterior predictive sampling
    
        Parameters
        ----------
        dist: Union[types.GeneratorType, pymc4.coroutine_model.Model]
            The 
        model_info: Mapping[str, Any]
            Either ``dist.model_info`` or 
            ``pymc4.coroutine_model.Model.default_model_info`` if ``dist`` is not a
            ``pymc4.courutine_model.Model`` instance.
        state: SamplingState
            The model's evaluation state.
    
        Returns
        -------
        model: Union[types.GeneratorType, pymc4.coroutine_model.Model]
            The original ``dist`` if it was not an observed ``Distribution`` or
            the ``Distribution`` with the changed ``batch_shape`` and observations
            set to ``None``.
    
        Raises
        ------
        EvaluationError
            When ``dist`` and its passed observed value don't have a consistent
            shape
        """
        dist = super().modify_distribution(dist, model_info, state)
        # We only modify the shape of Distribution instances that have observed
        # values
        if not isinstance(dist, Distribution):
            return dist
        scoped_name = scopes.variable_name(dist.name)

        observed_value = observed_value_in_evaluation(scoped_name, dist, state)
        if observed_value is None:
            return dist

        # We set the state's observed value to None to explicitly override
        # any previously given observed and at the same time, have the
        # scope_name added to the posterior_predictives set in
        # self.proceed_distribution
        state.observed_values[scoped_name] = None

        # We first check the TFP distribution's shape and compare it with the
        # observed_value's shape
        assert_observations_compatible_with_distribution_shape(scoped_name, observed_value, dist)

        # Now we get the broadcasted shape between the observed value and the distribution
        observed_shape = get_observed_tensor_shape(observed_value)
        dist_shape = dist._distribution.batch_shape + dist._distribution.event_shape
        new_dist_shape = tf.broadcast_static_shape(observed_shape, dist_shape)
        plate = new_dist_shape[: len(new_dist_shape) - len(dist_shape)]

        # Now we construct and return the same distribution but setting
        # observed to None and setting a batch_size that matches the result of
        # broadcasting the observed and distribution shape
        new_dist = type(dist)(
            name=dist.name, transform=dist.transform, observed=None, plate=plate, **dist.conditions,
        )
        return new_dist
