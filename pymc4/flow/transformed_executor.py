import copy

from pymc4 import scopes, distributions
from pymc4.distributions import abstract
from pymc4.distributions.abstract.transforms import JacobianPreference
from .executor import SamplingExecutor, EvaluationError


class TransformedSamplingExecutor(SamplingExecutor):
    def modify_distribution(self, dist, model_info, state):
        dist = super().modify_distribution(dist, model_info, state)
        if not isinstance(dist, abstract.Distribution):
            return dist
        # this will be mentioned later, that's the way to avoid loops
        if dist.transform is None:
            return dist
        transform = dist.transform
        scoped_name = scopes.variable_name(dist.name)
        transformed_scoped_name = scopes.variable_name(
            # double underscore stands for transform
            "__{}_{}".format(transform.name, dist.name)
        )
        if transformed_scoped_name in state.values or scoped_name in state.values:
            # We do not sample in this if branch

            # 0. do not allow ambiguity in state, make sure only one value is provided to compute logp
            if transformed_scoped_name in state.values and scoped_name in state.values:
                raise EvaluationError(
                    "Found both transformed and untransformed variables in the state: "
                    "'{} and '{}', but need exactly one".format(
                        scoped_name, transformed_scoped_name
                    )
                )

            def model():
                # 1. now compute all the variables: in the transformed and untransformed space
                if transformed_scoped_name in state.values:
                    transformed_value = state.values[transformed_scoped_name]
                    untransformed_value = transform.backward(transformed_value)
                else:
                    untransformed_value = state.values[scoped_name]
                    transformed_value = transform.forward(untransformed_value)
                # these lines below
                state.values[scoped_name] = untransformed_value
                state.values[transformed_scoped_name] = transformed_value
                # disable sampling and save cached results to store for yield dist

                # once we are done with variables we can yield the value in untransformed space
                # to the user and also increment the potential

                # Important:
                # I have no idea yet, how to make that beautiful.
                # Here we remove a transform to create another model that
                # essentially has an untransformed predefined value and the potential
                # needed to specify the logp properly
                dist_no_transform = copy.copy(dist)
                dist_no_transform.transform = None
                # If we do not set transform to None we will appear here again and again ending
                # up with an unclosed recursion

                # 2. here decide on logdet computation, this might be effective
                # with transformed value, but not with an untransformed one
                # this information is stored in transform.jacobian_preference class attribute
                if transform.jacobian_preference == JacobianPreference.Forward:
                    potential = transform.jacobian_log_det(untransformed_value)
                else:
                    potential = -transform.inverse_jacobian_log_det(transformed_value)
                yield distributions.Potential(potential)
                # 3. final return+yield will return untransformed_value
                # as it is stored in state.values
                # Note: we need yield here to make another checks on name duplicates, etc
                return (yield dist_no_transform)

        else:
            # we gonna sample here, but logp should be computed for the transformed space
            def model():
                # 0. as explained above we make a shallow copy of the distribution and remove the transform
                dist_no_transform = copy.copy(dist)
                dist_no_transform.transform = None
                # 1. sample a value, as we've checked there is no state provided
                # we need transform = None here not to get in a trouble
                # the return value is not yet user facing
                sampled_untransformed_value = yield dist_no_transform
                sampled_transformed_value = transform.forward(sampled_untransformed_value)
                # already stored untransformed value via yield
                # state.values[scoped_name] = sampled_untransformed_value
                state.values[transformed_scoped_name] = sampled_transformed_value
                # 2. increment the potential
                if transform.jacobian_preference == JacobianPreference.Forward:
                    potential = transform.jacobian_log_det(sampled_untransformed_value)
                else:
                    potential = -transform.inverse_jacobian_log_det(sampled_transformed_value)
                yield distributions.Potential(potential)
                # 3. return value to the user
                return sampled_untransformed_value

        # return the correct generator model instead of a distribution
        return model()
