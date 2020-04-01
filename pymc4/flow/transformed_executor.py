"""Execute graph in a transformed state.

Specifically, we wish to transform distributions whose support is bounded on
one or both sides to distributions that are supported for all real numbers.
"""
import functools


from types import GeneratorType
from typing import Mapping, Any, Generator
from pymc4 import scopes, distributions
from pymc4.distributions import distribution
from pymc4.distributions.transforms import JacobianPreference, Transform
from pymc4.flow.executor import (
    SamplingExecutor,
    EvaluationError,
    observed_value_in_evaluation,
    ModelType,
    SamplingState,
)


class TransformedSamplingExecutor(SamplingExecutor):
    """Perform inference in an unconstrained space."""

    def validate_state(self, state):
        """Validate that the model is not in a bad state."""
        return

    def modify_distribution(
        self, dist: ModelType, model_info: Mapping[str, Any], state: SamplingState
    ) -> ModelType:
        """Apply transformations to a ``ModelType``.
        
        This function calls the base class'
        :meth:`~pymc4.flow.executor.SamplingExecutor.modify_distribution` method on the supplied
        ``dist``. If the output of said call is not a :class:`pymc4.distributions.distribution.Distribution`
        instance, it is returned immediately. On the other hand, if the output is a
        :class:`pymc4.distributions.distribution.Distribution` instance, then it is passed to
        :func:`~.transform_dist_if_necessary` and the call's output is returned.

        Parameters
        ----------
        dist: ModelType
            Can be any type compatible with a pymc4 ``Model``, such as generator functions or
            ``Distribution`` instances. The type of ``dist`` is inspected to determine whether
            it should be passed onto :func:`~.transform_dist_if_necessary` or not.
        model_info: Mapping[str, Any]
            A dictionary of default model information parameters that is used to transform raw
            generator functions to :class:`~pymc4.model.Model` instances, by the base class'
            (:class:`~pymc4.flow.executor.SamplingExecutor`) ``modify_distribution`` method.
        state: pymc4.flow.executor.SamplingState
            The sampling state of the :meth:`~pymc4.flow.executor.SamplingExecutor.evaluate_model`
            flow.

        Returns
        -------
        modified: ModelType
            Either the output of the base class's
            :meth:`~pymc4.flow.executor.SamplingExecutor.modify_distribution` method, or the
            output of :func:`~.transform_dist_if_necessary`.
        """
        dist = super().modify_distribution(dist, model_info, state)
        if not isinstance(dist, distribution.Distribution):
            return dist

        return transform_dist_if_necessary(dist, state, allow_transformed_and_untransformed=True)


def make_untransformed_model(
    dist: distribution.Distribution, transform: Transform, state: SamplingState
) -> Generator:
    # we gonna sample here, but logp should be computed for the transformed space
    # 0. as explained above we indicate we already performed autotransform
    dist.model_info["autotransformed"] = True
    # 1. sample a value, as we've checked there is no state provided
    # we need `dist.model_info["autotransformed"] = True` here not to get in a trouble
    # the return value is not yet user facing
    sampled_untransformed_value = yield dist
    sampled_transformed_value = transform.forward(sampled_untransformed_value)
    # already stored untransformed value via yield
    # state.values[scoped_name] = sampled_untransformed_value
    transformed_scoped_name = scopes.transformed_variable_name(transform.name, dist.name)
    state.transformed_values[transformed_scoped_name] = sampled_transformed_value
    # 2. increment the potential
    if transform.jacobian_preference == JacobianPreference.Forward:
        potential_fn = functools.partial(
            transform.forward_log_det_jacobian, sampled_untransformed_value
        )
        coef = -1.0
    else:
        potential_fn = functools.partial(
            transform.inverse_log_det_jacobian, sampled_transformed_value
        )
        coef = 1.0
    yield distributions.Potential(potential_fn, coef=coef)
    # 3. return value to the user
    return sampled_untransformed_value


def make_transformed_model(
    dist: distribution.Distribution, transform: Transform, state: SamplingState
) -> Generator:
    # 1. now compute all the variables: in the transformed and untransformed space
    scoped_name = scopes.variable_name(dist.name)
    transformed_scoped_name = scopes.transformed_variable_name(transform.name, dist.name)
    state.untransformed_values[scoped_name] = transform.inverse(
        state.transformed_values[transformed_scoped_name]
    )
    # disable sampling and save cached results to store for yield dist

    # once we are done with variables we can yield the value in untransformed space
    # to the user and also increment the potential

    # Important:
    # I have no idea yet, how to make that beautiful.
    # Here we indicate the distribution is already autotransformed nto to get in the infinite loop
    dist.model_info["autotransformed"] = True

    # 2. here decide on logdet computation, this might be effective
    # with transformed value, but not with an untransformed one
    # this information is stored in transform.jacobian_preference class attribute
    # we postpone the computation of logdet as it might have some overhead
    if transform.jacobian_preference == JacobianPreference.Forward:
        potential_fn = functools.partial(
            transform.forward_log_det_jacobian, state.untransformed_values[scoped_name]
        )
        coef = -1.0
    else:
        potential_fn = functools.partial(
            transform.inverse_log_det_jacobian, state.transformed_values[transformed_scoped_name]
        )
        coef = 1.0
    yield distributions.Potential(potential_fn, coef=coef)
    # 3. final return+yield will return untransformed_value
    # as it is stored in state.values
    # Note: we need yield here to make another checks on name duplicates, etc
    return (yield dist)


def transform_dist_if_necessary(
    dist: distribution.Distribution, state: SamplingState, *, allow_transformed_and_untransformed
) -> ModelType:
    """Add the unbounded distribution to the executor's flow only if necessary.

    This function will inspect the provided distribution instance and state and determine whether
    it should intercept the normal coroutine's execution flow to add the unbounded representation
    of the distribution or not. If the unbounded representation must not be added, it returns the
    generator function that is outputed by :func:`~.make_untransformed_model`. On the other hand,
    if the unbounded representation of the ``dist`` instance must be added to the execution flow,
    it does so by returning the generator function that is given by calling
    :func:`~.make_transformed_model`.

    The regular execution flow will be intercepted to add the unbounded representation of ``dist``
    if ``dist`` is not an observed value and has a transformed scope name.

    Parameters
    ----------
    dist: pymc4.distributions.distribution.Distribution
        The original :class:`~pymc4.distributions.distribution.Distribution` instance that is
        inspected to decide whether to add its unbounded representation or not to the coroutine's
        execution flow.
    state: pymc4.flow.executor.SamplingState
        The sampling state of the :meth:`~pymc4.flow.executor.SamplingExecutor.evaluate_model`
        flow.
    allow_transformed_and_untransformed: bool
        If ``False``, the original instance's untransformed value is popped from the ``state``.
        Only the unbounded representation's value will be left in the ``transformed_values``
        attribute of ``state``.

    Returns
    -------
    model_maker: ModelType
        A coroutine that will yield the distribution's normal and unbounded representation
        values, along with a ``Potential`` with the appropriate ``log_det_jacobian`` value.
        Will either be the output of :func:`~.make_untransformed_model` or
        :func:`~.make_transformed_model` depending on whether the supplied ``dist`` must be
        unbounded or not.
    """
    if dist.transform is None or dist.model_info.get("autotransformed", False):
        return dist
    scoped_name = scopes.variable_name(dist.name)
    transform = dist.transform
    transformed_scoped_name = scopes.transformed_variable_name(transform.name, dist.name)
    if observed_value_in_evaluation(scoped_name, dist, state) is not None:
        # do not modify a distribution if it is observed
        # same for programmatically observed
        # but not for programmatically set to unobserved (when value is None)
        # but raise if we have transformed value passed in dict
        if transformed_scoped_name in state.transformed_values:
            raise EvaluationError(
                EvaluationError.OBSERVED_VARIABLE_IS_NOT_SUPPRESSED_BUT_ADDITIONAL_TRANSFORMED_VALUE_PASSED.format(
                    scoped_name, transformed_scoped_name
                )
            )
        if scoped_name in state.untransformed_values:
            raise EvaluationError(
                EvaluationError.OBSERVED_VARIABLE_IS_NOT_SUPPRESSED_BUT_ADDITIONAL_VALUE_PASSED.format(
                    scoped_name, scoped_name
                )
            )
        return dist

    if transformed_scoped_name in state.transformed_values:
        if (not allow_transformed_and_untransformed) and scoped_name in state.untransformed_values:
            state.untransformed_values.pop(scoped_name)
        return make_transformed_model(dist, transform, state)
    else:
        return make_untransformed_model(dist, transform, state)
