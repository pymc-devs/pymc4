import copy

from pymc4 import scopes, distributions
from pymc4.distributions import abstract
from pymc4.distributions.abstract.transforms import JacobianPreference
from pymc4.flow import SamplingExecutor


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
            if transformed_scoped_name in state.values:
                transformed_value = state.values[transformed_scoped_name]
                untransformed_value = transform.backward(transformed_value)
                state.values[scoped_name] = untransformed_value
                state.values[transformed_scoped_name] = transformed_value
            else:
                untransformed_value = state.values[scoped_name]
                transformed_value = transform.forward(untransformed_value)
                state.values[scoped_name] = untransformed_value
                state.values[transformed_scoped_name] = transformed_value

            def model():
                # I have no idea yet, how to make that beautiful
                # here I remove a transform to create another model that
                # essentially has an untransformed predefined value and the potential
                # needed to specify the logp properly
                dist_no_transform = copy.copy(dist)
                dist_no_transform.transform = None

                # these lines
                # > state.values[scoped_name] = untransformed_value
                # > state.values[transformed_scoped_name] = transformed_value
                # disable sampling and save cached results to store for yield dist
                # another step is to decide on logdet computation, this might be effective
                # with transformed value, but not with an untransformed one
                # this information is stored in transform.jacobian_preference class attribute
                if transform.jacobian_preference == JacobianPreference.Forward:
                    potential = transform.jacobian_log_det(untransformed_value)
                else:
                    potential = -transform.inverse_jacobian_log_det(transformed_value)
                yield distributions.Potential(potential)
                # will return untransformed_value
                # as it is stored in state.values
                return (yield dist_no_transform)

        else:
            # we gonna sample here, but logp should be computed for the transformed values
            def model():
                # as explained above we make a shallow copy of the distribution and remove the transform
                dist_no_transform = copy.copy(dist)
                dist_no_transform.transform = None
                # sample a value, as we've checked there is no state provided
                sampled_untransformed_value = yield dist_no_transform
                sampled_transformed_value = transform.forward(sampled_untransformed_value)
                # already stored
                # state.values[scoped_name] = sampled_untransformed_value
                state.values[transformed_scoped_name] = sampled_transformed_value

                if transform.jacobian_preference == JacobianPreference.Forward:
                    potential = transform.jacobian_log_det(sampled_untransformed_value)
                else:
                    potential = -transform.inverse_jacobian_log_det(sampled_transformed_value)
                yield distributions.Potential(potential)
                return sampled_untransformed_value

        return model()
