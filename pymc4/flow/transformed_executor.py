"""Execute graph in a transformed state.

Specifically, we wish to transform distributions whose support is bounded on
one or both sides to distributions that are supported for all real numbers.
"""
import functools
from pymc4 import scopes, distributions
from pymc4.distributions import distribution
from pymc4.distributions.transforms import JacobianPreference
from pymc4.flow.executor import SamplingExecutor, EvaluationError, observed_value_in_evaluation


class TransformedSamplingExecutor(SamplingExecutor):
    """Perform inference in an unconstrained space."""

    def validate_state(self, state):
        """Validate that the model is not in a bad state."""
        return

    def modify_distribution(self, dist, model_info, state):
        """Apply transformations to a distribution."""
        dist = super().modify_distribution(dist, model_info, state)
        if not isinstance(dist, distribution.Distribution):
            return dist
        scoped_name = scopes.variable_name(dist.name)

        if dist.transform is None or dist.model_info.get(  # do nothing else if no transform is set
            "autotransformed", False
        ):  # already autotransformed, do nothing else
            return dist
        transform = dist.transform
        transformed_scoped_name = scopes.variable_name(
            # double underscore stands for transform
            "__{}_{}".format(transform.name, dist.name)
        )
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
            # We do not sample in this if branch

            # 0. do not allow ambiguity in state, make sure only one value is provided to compute logp
            if (
                transformed_scoped_name in state.transformed_values
                and scoped_name in state.untransformed_values
            ):
                raise EvaluationError(
                    "Found both transformed and untransformed variables in the state: "
                    "'{} and '{}', but need exactly one".format(
                        scoped_name, transformed_scoped_name
                    )
                )

            def model():
                # 1. now compute all the variables: in the transformed and untransformed space
                if transformed_scoped_name in state.transformed_values:
                    transformed_value = state.transformed_values[transformed_scoped_name]
                    untransformed_value = transform.inverse(transformed_value)
                else:
                    untransformed_value = state.untransformed_values[scoped_name]
                    transformed_value = transform.forward(untransformed_value)
                # these lines below
                state.untransformed_values[scoped_name] = untransformed_value
                state.transformed_values[transformed_scoped_name] = transformed_value
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
                        transform.forward_log_det_jacobian, untransformed_value
                    )
                    coef = -1.0
                else:
                    potential_fn = functools.partial(
                        transform.inverse_log_det_jacobian, transformed_value
                    )
                    coef = 1.0
                yield distributions.Potential(potential_fn, coef=coef)
                # 3. final return+yield will return untransformed_value
                # as it is stored in state.values
                # Note: we need yield here to make another checks on name duplicates, etc
                return (yield dist)

        else:
            # we gonna sample here, but logp should be computed for the transformed space
            def model():
                # 0. as explained above we indicate we already performed autotransform
                dist.model_info["autotransformed"] = True
                # 1. sample a value, as we've checked there is no state provided
                # we need `dist.model_info["autotransformed"] = True` here not to get in a trouble
                # the return value is not yet user facing
                sampled_untransformed_value = yield dist
                sampled_transformed_value = transform.forward(sampled_untransformed_value)
                # already stored untransformed value via yield
                # state.values[scoped_name] = sampled_untransformed_value
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

        # return the correct generator model instead of a distribution
        return model()
