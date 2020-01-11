"""Execute graph with test values to extract a model's meta-information.

Specifically, we wish to extract:

- All variable's core shapes
- All observed, deterministic, and unobserved variables (both transformed and
untransformed.
"""
from typing import Tuple, Any
from pymc4 import scopes
from pymc4.distributions import distribution
from pymc4.flow.executor import (
    SamplingState,
    EvaluationError,
    observed_value_in_evaluation,
    assert_values_compatible_with_distribution,
)
from pymc4.flow.transformed_executor import TransformedSamplingExecutor


class MetaExecutor(TransformedSamplingExecutor):
    """Perform inference in an unconstrained space."""

    def proceed_distribution(
        self, dist: distribution.Distribution, state: SamplingState
    ) -> Tuple[Any, SamplingState]:
        # TODO: docs
        if dist.is_anonymous:
            raise EvaluationError("Attempting to create an anonymous Distribution")
        scoped_name = scopes.variable_name(dist.name)
        if scoped_name in state.distributions or scoped_name in state.deterministics:
            raise EvaluationError(
                "Attempting to create a duplicate variable {!r}, "
                "this may happen if you forget to use `pm.name_scope()` when calling same "
                "model/function twice without providing explicit names. If you see this "
                "error message and the function being called is not wrapped with "
                "`pm.model`, you should better wrap it to provide explicit name for this model".format(
                    scoped_name
                )
            )
        if scoped_name in state.observed_values or dist.is_observed:
            observed_variable = observed_value_in_evaluation(scoped_name, dist, state)
            if observed_variable is None:
                # None indicates we pass None to the state.observed_values dict,
                # might be posterior predictive or programmatically override to exchange observed variable to latent
                if scoped_name not in state.untransformed_values:
                    # posterior predictive
                    return_value = state.untransformed_values[scoped_name] = dist.sample()
                else:
                    # replace observed variable with a custom one
                    return_value = state.untransformed_values[scoped_name]
                # We also store the name in posterior_predictives just to keep
                # track of the variables used in posterior predictive sampling
                state.posterior_predictives.add(scoped_name)
                state.observed_values.pop(scoped_name)
            else:
                if scoped_name in state.untransformed_values:
                    raise EvaluationError(
                        EvaluationError.OBSERVED_VARIABLE_IS_NOT_SUPPRESSED_BUT_ADDITIONAL_VALUE_PASSED.format(
                            scoped_name
                        )
                    )
                assert_values_compatible_with_distribution(scoped_name, observed_variable, dist)
                return_value = state.observed_values[scoped_name] = observed_variable
        elif scoped_name in state.untransformed_values:
            return_value = state.untransformed_values[scoped_name]
        else:
            return_value = state.untransformed_values[scoped_name] = dist.sample()
        state.distributions[scoped_name] = dist
        return return_value, state
