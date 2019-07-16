import types
from typing import Any, Tuple, Dict, Union, List
import collections
import itertools
import pymc4 as pm
from pymc4 import coroutine_model
from pymc4 import scopes
from pymc4 import utils
from pymc4.distributions import abstract


ModelType = Union[types.GeneratorType, coroutine_model.Model]


class EvaluationError(RuntimeError):
    # common error messages
    OBSERVED_VARIABLE_IS_NOT_SUPPRESSED_BUT_ADDITIONAL_VALUE_PASSED = (
        "Attempting to evaluate a model with both "
        "observed and unobserved values provided "
        "what requires to choose what value to actually yield. "
        "To remove this error you either need to add `observed={{{0!r}: None}}` "
        "or remove {0!r} from untransformed values."
    )
    OBSERVED_VARIABLE_IS_NOT_SUPPRESSED_BUT_ADDITIONAL_TRANSFORMED_VALUE_PASSED = (
        "Attempting to evaluate a model with both "
        "observed and unobserved values provided "
        "what requires to choose what value to actually yield. "
        "To remove this error you either need to add `observed={{{0!r}: None}}` "
        "or remove {1!r} from transformed values."
    )
    ...


class StopExecution(StopIteration):
    NOT_HELD_ERROR_MESSAGE = """
    for some reason outer scope (control flow) may silence an exception in `yield`
    # try:
    #     yield something_bad
    # except:
    #     pass
    # ...
    in that case `gen.throw(error)` will go silently, but we can keep track of it and capture
    silent behaviour and create an error in an model evaluation scope, not in control flow
    this error handling should prevent undefined behaviour. However, if the control flow does not have
    error handling stuff, the Exception is injected directly to the place where it was occurred
    if we ever appear at this particular point we may by whatever deep in call stack of `evaluate_model`
    and therefore bypass this error up the stack.
    """


class EarlyReturn(StopIteration):
    ...


class SamplingState(object):
    __slots__ = (
        "transformed_values",
        "untransformed_values",
        "observed_values",
        "all_values",
        "distributions",
        "potentials",
    )

    def __init__(
        self,
        transformed_values: Dict[str, Any] = None,
        untransformed_values: Dict[str, Any] = None,
        observed_values: Dict[str, Any] = None,
        distributions: Dict[str, abstract.Distribution] = None,
        potentials: List[abstract.Potential] = None,
    ):
        # verbose __init__
        if transformed_values is None:
            transformed_values = dict()
        else:
            transformed_values = transformed_values.copy()
        if untransformed_values is None:
            untransformed_values = dict()
        else:
            untransformed_values = untransformed_values.copy()
        if observed_values is None:
            observed_values = dict()
        else:
            observed_values = observed_values.copy()
        if distributions is None:
            distributions = dict()
        else:
            distributions = distributions.copy()
        if potentials is None:
            potentials = list()
        else:
            potentials = potentials.copy()
        self.transformed_values = transformed_values
        self.untransformed_values = untransformed_values
        self.observed_values = observed_values
        self.all_values = collections.ChainMap(
            self.untransformed_values, self.transformed_values, self.observed_values
        )
        self.distributions = distributions
        self.potentials = potentials

    def collect_log_prob(self):
        return sum(
            itertools.chain(
                (dist.log_prob(self.all_values[name]) for name, dist in self.distributions.items()),
                (p.value for p in self.potentials),
            )
        )

    def __repr__(self):
        # display keys only
        untransformed_values = list(self.untransformed_values)
        transformed_values = list(self.transformed_values)
        observed_values = list(self.observed_values)
        # format like dist:name
        distributions = [
            "{}:{}".format(d.__class__.__name__, k) for k, d in self.distributions.items()
        ]
        # be less verbose here
        num_potentials = len(self.potentials)
        indent = 4 * " "
        return (
            "{}(\n"
            + indent
            + "untransformed_values: {}\n"
            + indent
            + "transformed_values: {}\n"
            + indent
            + "observed_values: {}\n"
            + indent
            + "distributions: {}\n"
            + indent
            + "num_potentials={})"
        ).format(
            self.__class__.__name__,
            untransformed_values,
            transformed_values,
            observed_values,
            distributions,
            num_potentials,
        )

    @classmethod
    def from_values(cls, values: Dict[str, Any] = None, observed_values: Dict[str, Any] = None):
        if values is None:
            return cls(observed_values=observed_values)
        transformed_values = dict()
        untransformed_values = dict()
        # split by `nest/name` or `nest/__transform_name`
        for fullname in values:
            namespec = utils.NameParts.from_name(fullname)
            if namespec.is_transformed:
                transformed_values[fullname] = values[fullname]
            else:
                untransformed_values[fullname] = values[fullname]
        return cls(transformed_values, untransformed_values, observed_values)

    def clone(self):
        return self.__class__(
            transformed_values=self.transformed_values,
            untransformed_values=self.untransformed_values,
            observed_values=self.observed_values,
            distributions=self.distributions,
            potentials=self.potentials,
        )

    def as_sampling_state(self):
        """Create a sampling state that should me used within MCMC sampling.

        There are some principles that hold for the state.

            1. Check there is at least one distribution
            2. Check all transformed distributions are autotransformed
            3. Remove untransformed values if transformed are present
            4. Remove all other irrelevant values
        """
        if not self.distributions:
            raise TypeError(
                "No distributions found in the state. "
                "the model you evaluated is empty and does not yield any PyMC4 distribution"
            )
        untransformed_values = dict()
        transformed_values = dict()
        observed_values = dict()
        for name, dist in self.distributions.items():
            namespec = utils.NameParts.from_name(name)
            if dist.transform is not None and name not in self.observed_values:
                transformed_namespec = namespec.replace_transform(dist.transform.name)
                if transformed_namespec.full_original_name not in self.transformed_values:
                    raise TypeError(
                        "Transformed value {!r} is not found for {} distribution with name {!r}. "
                        "You should evaluate the model using the transformed executor to get "
                        "the correct sampling state.".format(
                            transformed_namespec.full_original_name, dist, name
                        )
                    )
                else:
                    transformed_values[
                        transformed_namespec.full_original_name
                    ] = self.transformed_values[transformed_namespec.full_original_name]
            else:
                if name in self.observed_values:
                    observed_values[name] = self.observed_values[name]
                elif name in self.untransformed_values:
                    untransformed_values[name] = self.untransformed_values[name]
                else:
                    raise TypeError(
                        "{} distribution with name {!r} does not have the corresponding value "
                        "in the state. This may happen if the current "
                        "state was modified in the wrong way."
                    )
        return self.__class__(
            transformed_values=transformed_values,
            untransformed_values=untransformed_values,
            observed_values=observed_values,
        )


# when we make changes in a subclass we usually call super() that will require self to be present in signature
# therefore it is convenient to disable inspection about static methods

# noinspection PyMethodMayBeStatic
class SamplingExecutor(object):
    """
    Base untransformed executor.

    This executor performs model evaluation in the untransformed space. Class structure is convenient since its
    subclass :class:`TransformedSamplingExecutor` will reuse some parts from parent class and extending functionality.
    """

    def validate_return_object(self, return_object: Any):
        if isinstance(
            return_object, (coroutine_model.Model, types.GeneratorType, abstract.Potential)
        ):
            raise EvaluationError(
                "Return values should not contain instances of "
                "apm.coroutine_model.Model`, "
                "`types.GeneratorType` "
                "and `pm.distributions.Potential`. "
                "To fix the error you should change the return statement to something like\n"
                "    ..."
                "    return (yield variable)"
            )

    def validate_return_value(self, return_value: Any):
        pm.utils.map_nested(self.validate_return_object, return_value)

    def evaluate_model(
        self,
        model: ModelType,
        *,
        state: SamplingState = None,
        _validate_state: bool = True,
        values: Dict[str, Any] = None,
        observed: Dict[str, Any] = None,
    ) -> Tuple[Any, SamplingState]:
        # this will be dense with comments as all interesting stuff is composed in here

        # 1) we need to check for state or generate one

        #   in a state we might have:
        #   - return values for distributions/models

        # we will modify this state with:
        #   - distributions, used to calculate logp correctly
        #   - potentials, same purpose
        #   - values in case we sample from a distribution

        # All this is subclassed and implemented in Sampling Executor. This class tries to be
        # as general as possible, just to restrict the imagination and reduce complexity.

        if state is None:
            state = self.new_state(values=values, observed=observed)
        else:
            if values or observed:
                raise ValueError("Provided arguments along with not empty state")
        if _validate_state:
            self.validate_state(state)
        # 2) we can proceed 2 typed of models:
        #   - generator object that yields other model-like objects
        #   - Model objects that come up with additional user provided information

        # some words about "user provided information"
        # they are:
        #   - keep_auxiliary
        #       in pymc4 it is model is some generator, that yields distributions and possibly returns a value
        #       at the point of posterior sampling we may want to skip some computations. This is
        #       mainly developer feature that allows to implement compositional distributions like Horseshoe and
        #       at posterior predictive sampling omit some redundant parts of computational graph
        #   - keep_return
        #       the return value of generator will be saved in state.untransformed_values if this set to True.
        #   - observed
        #       the observed variable(s) for the given model. There are some restrictions on how do we provide
        #       observed variables.
        #       - Every observed variable should have a name either explicitly or implicitly.
        #           The explicit way to provide an observed variable is to use `observed` keyword in a Distribution API
        #           such as `Dist(..., observed=value)`. Observed variables are treated carefully in evaluation
        #           what will be covered a bit later. The other way, a bit less explicit, is to provide a name
        #           for the observed variable is creating a dictionary Dict[str, Any] that maps node names to
        #           the observed values and provide it to the evaluator.
        #       - Every distribution has to have only one value or none of them: either observed or (un)transformed one.
        #           This constraint removes undefined choice and the need to guess for the intent.
        #       Consequently, the above conventions create interesting situations that may appear in API usage.
        #       As mentioned above, a distribution has to have only one value provided or none. Therefore we have to
        #       perform a check at some point to make sure no ambiguous situations happen. The accepted cases should
        #       then contain:
        #       1. observed value is provided in `Dist(..., observed=value)`. This is the regular usage of PyMC4
        #       2. observed value provided in `Dist(..., observed=value)` is suppressed. The case we perform
        #           posterior/prior predictive checks. If the value is not suppressed, the executor will
        #           yield the old value without sampling.
        #       3. observed value provided in `Dist(..., observed=value)` is replaced with new data. The common case
        #           of model being reused to fit for different datasets
        #       4. unobserved value is provided in executor directly to replace an observed value.
        #           An advanced usage of PyMC4 where we change the set of observed nodes (not variables)
        #
        #       The above 4 cases should work with MCMC sampling engine and provide the correct log probability
        #       computation from PyMC4 side. We should now recall that all the computation of log probability
        #       is done in the transformed space, where parameters lie in Rn without any constraints.
        #
        #       The tempting questing is "what happens with the observed variables?". Without careful treatment
        #       we would compute log probability for the transformed space if `Dist` in `Dist(..., observed=value)`
        #       is bounded. But this is not required and redundant, the observed variable does not violate the bounds
        #       and is not adjusted in MCMC. Therefore we should take care and not autotransfrom them in the transformed
        #       Executor.
        #
        #       Some other issues may happen if we omit checks proceeding cases 1-4
        #       Case 1. Nothing special here. We just make sure not to see the same value again in unobserved that is
        #           probably a mistake.
        #       Case 2. How do we know we need forward sample a particular observed node? The solution is to provide a
        #           convention that passing `observed={"observed/variable/name": None}` suppresses an observed set as
        #           `Dist(..., observed=value)` and instructs to sample from this distribution.
        #       Case 3. That's probably explicit to pass `observed={"observed/variable/name": new_value}` to executor
        #           and ask to replace the old observed
        #       Case 4. Suppose we provide an observed value for a specific variable using `Dist(..., observed=value)`.
        #           What should happen if we accidentally pass (un)transformed value to the executor with the same name?
        #           Do we want to replace the observed value with a new one but not observed or it is a typo/mistake?
        #           It may be very hard to track the intent except having a convention that we want to override the
        #           value in here and perform inference on it. My (@ferrine'e) guess is that convention is very unsafe
        #           and implicit, therefore I propose to additionally provide an intent suppressing the observed
        #           variable by name. Actual code should contain `observed={"old/observed/variable/name": None}` passed
        #           along with `values={"old/observed/variable/name": new_inference_value}` to the executor. We can be
        #           sure there is no mistake and safely replace "old/observed/variable/name" with an unobserved one.
        #
        #       You may now see, that logic becomes quite complicated once we care about the details and common
        #       use cases. So far we figured out what checks are to be performed, the next question is at what time they
        #       are performed. In most cases we only get enough information at the time we actually see the node, not
        #       before the execution, therefore there is not choice but to put all the execution validation at runtime.
        #       These checks differ a bit for the transformed and untransformed variables as for the debug purposes
        #       (we want to get the initial sampling state but bored in manually applying the transform) a provided
        #       value for bounded distribution may be either in transformed or untransformed space. So far validation
        #       logic is spread among different places, it probably worth unifying we way we validate execution state.

        if isinstance(model, abstract.Distribution):
            # usually happens when
            #   pm.evaluate_model(pm.distributions.Normal("n", 0, 1))
            # is called
            # we instead wrap it in an anonymous generator with single yield
            # that sets the correct namespaces
            # without it, we obtain values={"n/n": ...}
            # However, we disallow return statements that are models
            _model_ref = model
            model = (lambda: (yield _model_ref))()
        if isinstance(model, coroutine_model.Model):
            model_info = model.model_info
            try:
                control_flow = self.prepare_model_control_flow(model, model_info, state)
            except EarlyReturn as e:
                return e.args
        else:
            if not isinstance(model, types.GeneratorType):
                raise StopExecution(
                    "Attempting to call `evaluate_model` on a "
                    "non model-like object {}. Supported types are "
                    "`types.GeneratorType` and `pm.coroutine_model.Model`".format(type(model))
                )
            control_flow = model
            model_info = coroutine_model.Model.default_model_info
        return_value = None
        while True:
            try:
                with model_info["scope"]:
                    dist = control_flow.send(return_value)
                    if not isinstance(
                        dist, (types.GeneratorType, coroutine_model.Model, abstract.Potential)
                    ):
                        # prohibit any unknown type
                        error = EvaluationError(
                            "Type {} can't be processed in evaluation".format(type(dist))
                        )
                        control_flow.throw(error)
                        raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
                    # dist is a clean, known type

                    if isinstance(dist, abstract.Distribution):
                        dist = self.modify_distribution(dist, model_info, state)
                    if isinstance(dist, abstract.Potential):
                        state.potentials.append(dist)
                        return_value = dist
                    elif isinstance(dist, abstract.Distribution):
                        try:
                            return_value, state = self.proceed_distribution(dist, state)
                        except EvaluationError as error:
                            control_flow.throw(error)
                            raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
                    elif isinstance(dist, (coroutine_model.Model, types.GeneratorType)):
                        return_value, state = self.evaluate_model(
                            dist, state=state, _validate_state=False
                        )
                    else:
                        error = EvaluationError(
                            "Type {} can't be processed in evaluation. This error may appear "
                            "due to wrong implementation, please submit a bug report to "
                            "https://github.com/pymc-devs/pymc4/issues".format(type(dist))
                        )
                        control_flow.throw(error)
                        raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
            except StopExecution:
                # for some reason outer scope (control flow) may silence an exception in `yield`
                # try:
                #     yield something_bad
                # except:
                #     pass
                # ...
                # in that case `gen.throw(error)` will go silently, but we can keep track of it and capture
                # silent behaviour and create an error in an model evaluation scope, not in control flow
                # this error handling should prevent undefined behaviour. However, if the control flow does not have
                # error handling stuff, the Exception is injected directly to the place where it was occurred
                # if we ever appear at this particular point we may by whatever deep in call stack of `evaluate_model`
                # and therefore bypass this error up the stack
                # -----
                # this message with little modifications will appear in exception
                control_flow.close()
                raise
            except EarlyReturn as e:
                # for some reason we may raise it within model evaluation,
                # e.g. in self.proceed_distribution
                return e.args
            except StopIteration as stop_iteration:
                self.validate_return_value(stop_iteration.args[:1])
                return_value, state = self.finalize_control_flow(stop_iteration, model_info, state)
                break
        return return_value, state

    __call__ = evaluate_model

    def new_state(self, values: Dict[str, Any] = None, observed: Dict[str, Any] = None):
        return SamplingState.from_values(values=values, observed_values=observed)

    def validate_state(self, state):
        if state.transformed_values:
            raise ValueError(
                "untransformed executor should not contain "
                "transformed variables but found {}".format(set(state.transformed_values))
            )

    def modify_distribution(
        self, dist: abstract.Distribution, model_info: Dict[str, Any], state: SamplingState
    ):
        return dist

    def proceed_distribution(self, dist: abstract.Distribution, state: SamplingState):
        if dist.is_anonymous:
            raise EvaluationError("Attempting to create an anonymous Distribution")
        scoped_name = scopes.variable_name(dist.name)
        if scoped_name in state.distributions:
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
                state.observed_values.pop(scoped_name)
            else:
                if scoped_name in state.untransformed_values:
                    raise EvaluationError(
                        EvaluationError.OBSERVED_VARIABLE_IS_NOT_SUPPRESSED_BUT_ADDITIONAL_VALUE_PASSED.format(
                            scoped_name
                        )
                    )
                return_value = state.observed_values[scoped_name] = observed_variable
        elif scoped_name in state.untransformed_values:
            return_value = state.untransformed_values[scoped_name]
        else:
            return_value = state.untransformed_values[scoped_name] = dist.sample()
        state.distributions[scoped_name] = dist
        return return_value, state

    def prepare_model_control_flow(
        self, model: coroutine_model.Model, model_info: Dict[str, Any], state: SamplingState
    ):
        control_flow: types.GeneratorType = model.control_flow()
        model_name = model_info["name"]
        if model_name is None and model_info["keep_return"]:
            error = EvaluationError(
                "Attempting to create unnamed return variable when `keep_return` is set to True"
            )
            control_flow.throw(error)
            control_flow.close()
            raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
        return_name = scopes.variable_name(model_name)
        if not model_info["keep_auxiliary"] and return_name in state.untransformed_values:
            raise EarlyReturn(state.untransformed_values[model_name], state)
        return control_flow

    def finalize_control_flow(
        self, stop_iteration: StopIteration, model_info: Dict[str, Any], state: SamplingState
    ):
        if stop_iteration.args:
            return_value = stop_iteration.args[0]
        else:
            return_value = None
        if return_value is not None and model_info["keep_return"]:
            # we should filter out allowed return types, but this is totally backend
            # specific and should be determined at import time.
            return_name = scopes.variable_name(model_info["name"])
            state.untransformed_values[return_name] = return_value
        return return_value, state


def observed_value_in_evaluation(
    scoped_name: str, dist: abstract.Distribution, state: SamplingState
):
    return state.observed_values.get(scoped_name, dist.model_info["observed"])
