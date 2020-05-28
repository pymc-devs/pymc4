import types
from typing import Any, Tuple, Dict, Union, List, Optional, Set, Mapping
from collections import ChainMap
import itertools

import tensorflow as tf

import pymc4 as pm
from pymc4 import coroutine_model
from pymc4 import scopes
from pymc4 import utils
from pymc4.distributions import distribution


ModelType = Union[types.GeneratorType, coroutine_model.Model]
MODEL_TYPES = (types.GeneratorType, coroutine_model.Model)
MODEL_POTENTIAL_AND_DETERMINISTIC_TYPES = (
    types.GeneratorType,
    coroutine_model.Model,
    distribution.Potential,
    distribution.Deterministic,
)


def _chain_map_iter(self):
    """Keep ordering of maps on Python3.6.

    See https://bugs.python.org/issue32792

    Once Python3.6 is not supported, this can be deleted.
    """
    d = {}
    for mapping in reversed(self.maps):
        d.update(mapping)  # reuses stored hash values if possible
    return iter(d)


ChainMap.__iter__ = _chain_map_iter  # type: ignore


class EvaluationError(RuntimeError):
    """Errors that happen while a pymc4 model is evaluated."""

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
    INCOMPATIBLE_VALUE_AND_DISTRIBUTION_SHAPE = (
        "The values supplied to the distribution {0!r} are not consistent "
        "with the distribution's shape (dist_shape).\n"
        "dist_shape = batch_shape + event_shape = {1!r}\n"
        "Supplied values shape = {2!r}.\n"
        "A values array is considered to have a consistent shape with the "
        "distribution if two conditions are met.\n"
        "1) It has a greater or equal number of dimensions when compared to the "
        "distribution (len(value.shape) >= len(dist_shape))\n"
        "2) The values shape is compatible with the "
        "distribution's shape: "
        "dist_shape.is_compatible_with("
        "    value_shape[(len(values.shape) - len(dist_shape)):]"
        ")"
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


class SamplingState:
    __slots__ = (
        "transformed_values",
        "untransformed_values",
        "observed_values",
        "posterior_predictives",
        "all_values",
        "all_unobserved_values",
        "distributions",
        "potentials",
        "deterministics",
    )

    def __init__(
        self,
        transformed_values: Dict[str, Any] = None,
        untransformed_values: Dict[str, Any] = None,
        observed_values: Dict[str, Any] = None,
        distributions: Dict[str, distribution.Distribution] = None,
        potentials: List[distribution.Potential] = None,
        deterministics: Dict[str, Any] = None,
        posterior_predictives: Optional[Set[str]] = None,
    ) -> None:
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
        if deterministics is None:
            deterministics = dict()
        else:
            deterministics = deterministics.copy()
        if posterior_predictives is None:
            posterior_predictives = set()
        else:
            posterior_predictives = posterior_predictives.copy()
        self.transformed_values = transformed_values
        self.untransformed_values = untransformed_values
        self.observed_values = observed_values
        self.all_values = ChainMap(
            self.untransformed_values, self.transformed_values, self.observed_values
        )
        self.all_unobserved_values = ChainMap(self.transformed_values, self.untransformed_values)
        self.distributions = distributions
        self.potentials = potentials
        self.deterministics = deterministics
        self.posterior_predictives = posterior_predictives

    def collect_log_prob_elemwise(self):
        return itertools.chain(
            (dist.log_prob(self.all_values[name]) for name, dist in self.distributions.items()),
            (p.value for p in self.potentials),
        )

    def collect_log_prob(self):
        return sum(map(tf.reduce_sum, self.collect_log_prob_elemwise()))

    def collect_unreduced_log_prob(self):
        return sum(self.collect_log_prob_elemwise())

    def __repr__(self):
        # display keys only
        untransformed_values = list(self.untransformed_values)
        transformed_values = list(self.transformed_values)
        observed_values = list(self.observed_values)
        deterministics = list(self.deterministics)
        posterior_predictives = list(self.posterior_predictives)
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
            + "num_potentials={}\n"
            + indent
            + "deterministics: {}\n"
            + indent
            + "posterior_predictives: {})"
        ).format(
            self.__class__.__name__,
            untransformed_values,
            transformed_values,
            observed_values,
            distributions,
            num_potentials,
            deterministics,
            posterior_predictives,
        )

    @classmethod
    def from_values(
        cls, values: Dict[str, Any] = None, observed_values: Dict[str, Any] = None
    ) -> "SamplingState":
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

    def clone(self) -> "SamplingState":
        return self.__class__(
            transformed_values=self.transformed_values,
            untransformed_values=self.untransformed_values,
            observed_values=self.observed_values,
            distributions=self.distributions,
            potentials=self.potentials,
            deterministics=self.deterministics,
            posterior_predictives=self.posterior_predictives,
        )

    def as_sampling_state(self) -> "Tuple[SamplingState, List[str]]":
        """Create a sampling state that should be used within MCMC sampling.

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
        need_to_transform_after = list()
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
                    need_to_transform_after.append(transformed_namespec.full_untransformed_name)
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
        return (
            self.__class__(
                transformed_values=transformed_values,
                untransformed_values=untransformed_values,
                observed_values=observed_values,
            ),
            need_to_transform_after,
        )


# when we make changes in a subclass we usually call super() that will require self to be present in signature
# therefore it is convenient to disable inspection about static methods

# noinspection PyMethodMayBeStatic
class SamplingExecutor:
    """Base untransformed executor.

    This executor performs model evaluation in the untransformed space. Class structure is convenient since its
    subclass :class:`TransformedSamplingExecutor` will reuse some parts from parent class and extending functionality.
    """

    def validate_return_object(self, return_object: Any):
        if isinstance(return_object, MODEL_POTENTIAL_AND_DETERMINISTIC_TYPES):
            raise EvaluationError(
                "Return values should not contain instances of "
                "a `pm.coroutine_model.Model`, "
                "`types.GeneratorType`, "
                "`pm.distributions.Deterministic`, "
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
        sample_shape: Union[int, Tuple[int], tf.TensorShape] = (),
    ) -> Tuple[Any, SamplingState]:
        # this will be dense with comments as all interesting stuff is composed in here

        # 1) we need to check for state or generate

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
        # 2) we can proceed with 2 types of models:
        #   1. generator object that yields other model-like objects
        #   2. Model objects that come up with additional user provided information

        # some words about "user provided information"
        # they are:
        #   - keep_auxiliary
        #       in pymc4 it is model is some generator, that yields distributions and possibly returns a value
        #       at the point of posterior sampling we may want to skip some computations. This is
        #       mainly developer feature that allows to implement compositional distributions like Horseshoe and
        #       at posterior predictive sampling omit some redundant parts of computational graph
        #   - keep_return
        #       the return value of generator will be saved in state.deterministics if this set to True.
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
        #       The tempting question is "what happens with the observed variables?". Without careful treatment
        #       we would compute log probability for the transformed space if `Dist` in `Dist(..., observed=value)`
        #       is bounded. But this is not required and redundant, the observed variable does not violate the bounds
        #       and is not adjusted in MCMC. Therefore we should take care and not autotransfrom them in the transformed
        #       Executor.
        #
        #       Some other issues may happen if we omit checks proceeding cases 1-4
        #       Case 1. Nothing special here. We just make sure not to see the same value again in unobserved that is
        #           probably a mistake.
        #       Case 2. How do we know we need to forward sample a particular observed node? The solution is to provide a
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

        if isinstance(model, distribution.Distribution):
            # usually happens when
            #   pm.evaluate_model(pm.distributions.Normal("n", 0, 1))
            # is called
            # we instead wrap it in an anonymous generator with single yield
            # that sets the correct namespaces
            # without it, we obtain values={"n/n": ...}
            # However, we disallow return statements that are models
            _model_ref = model
            model = (lambda: (yield _model_ref))()  # type: ignore
        if isinstance(model, coroutine_model.Model):
            model_info = model.model_info
            try:
                control_flow = self.prepare_model_control_flow(model, model_info, state)
            except EarlyReturn as e:
                return e.args, state
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
                    if not isinstance(dist, MODEL_POTENTIAL_AND_DETERMINISTIC_TYPES):
                        # prohibit any unknown type
                        error = EvaluationError(
                            "Type {} can't be processed in evaluation".format(type(dist))
                        )
                        control_flow.throw(error)
                        raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error

                    # dist is a clean, known type from here on

                    # If distribution, potentially transform it
                    if isinstance(dist, distribution.Distribution):
                        dist = self.modify_distribution(dist, model_info, state)
                    if isinstance(dist, distribution.Potential):
                        state.potentials.append(dist)
                        return_value = dist
                    elif isinstance(dist, distribution.Deterministic):
                        try:
                            return_value, state = self.proceed_deterministic(dist, state)
                        except EvaluationError as error:
                            control_flow.throw(error)
                            raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
                    elif isinstance(dist, distribution.Distribution):
                        try:
                            return_value, state = self.proceed_distribution(
                                dist, state, sample_shape=sample_shape
                            )
                        except EvaluationError as error:
                            control_flow.throw(error)
                            raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
                    elif isinstance(dist, MODEL_TYPES):
                        return_value, state = self.evaluate_model(
                            dist, state=state, _validate_state=False, sample_shape=sample_shape
                        )
                    else:
                        err = EvaluationError(
                            "Type {} can't be processed in evaluation. This error may appear "
                            "due to wrong implementation, please submit a bug report to "
                            "https://github.com/pymc-devs/pymc4/issues".format(type(dist))
                        )
                        control_flow.throw(err)
                        raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from err
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
                return e.args, state
            except StopIteration as stop_iteration:
                self.validate_return_value(stop_iteration.args[:1])
                return self.finalize_control_flow(stop_iteration, model_info, state)
        return return_value, state

    __call__ = evaluate_model

    def new_state(
        self, values: Dict[str, Any] = None, observed: Dict[str, Any] = None
    ) -> SamplingState:
        return SamplingState.from_values(values=values, observed_values=observed)

    def validate_state(self, state):
        if state.transformed_values:
            raise ValueError(
                "untransformed executor should not contain "
                "transformed variables but found {}".format(set(state.transformed_values))
            )

    def modify_distribution(
        self, dist: ModelType, model_info: Mapping[str, Any], state: SamplingState
    ) -> ModelType:
        return dist

    def proceed_distribution(
        self,
        dist: distribution.Distribution,
        state: SamplingState,
        sample_shape: Union[int, Tuple[int], tf.TensorShape] = None,
    ) -> Tuple[Any, SamplingState]:
        # TODO: docs
        if dist.is_anonymous:
            raise EvaluationError("Attempting to create an anonymous Distribution")
        scoped_name = scopes.variable_name(dist.name)
        if scoped_name is None:
            raise EvaluationError("Attempting to create an anonymous Distribution")

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
                    if dist.is_root:
                        return_value = state.untransformed_values[scoped_name] = dist.sample(
                            sample_shape=sample_shape
                        )
                    else:
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
            if dist.is_root:
                return_value = state.untransformed_values[scoped_name] = dist.sample(
                    sample_shape=sample_shape
                )
            else:
                return_value = state.untransformed_values[scoped_name] = dist.sample()
        state.distributions[scoped_name] = dist
        return return_value, state

    def proceed_deterministic(
        self, deterministic: distribution.Deterministic, state: SamplingState
    ) -> Tuple[Any, SamplingState]:
        # TODO: docs
        if deterministic.is_anonymous:
            raise EvaluationError("Attempting to create an anonymous Deterministic")
        scoped_name = scopes.variable_name(deterministic.name)
        if scoped_name is None:
            raise EvaluationError("Attempting to create an anonymous Deterministic")
        if scoped_name in state.distributions or scoped_name in state.deterministics:
            raise EvaluationError(
                "Attempting to create a duplicate deterministic {!r}, "
                "this may happen if you forget to use `pm.name_scope()` when calling same "
                "model/function twice without providing explicit names. If you see this "
                "error message and the function being called is not wrapped with "
                "`pm.model`, you should better wrap it to provide explicit name for this model".format(
                    scoped_name
                )
            )
        state.deterministics[scoped_name] = return_value = deterministic.get_value()
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
            return_name = scopes.variable_name(model_info["name"])
            if return_name is None:
                raise AssertionError(
                    "Attempting to create unnamed return variable *after* making a check"
                )

            state.deterministics[return_name] = return_value
        return return_value, state


def observed_value_in_evaluation(
    scoped_name: str, dist: distribution.Distribution, state: SamplingState
):
    return state.observed_values.get(scoped_name, dist.model_info["observed"])


def assert_values_compatible_with_distribution(
    scoped_name: str, values: Any, dist: distribution.Distribution
) -> None:
    """Assert if the Distribution's shape is compatible with the supplied values.
    
    A distribution's shape, ``dist_shape``, is made up by the sum of
    the ``batch_shape`` and the ``event_shape``.

    A value is considered to have a consistent shape with the distribution if
    two conditions are met.
    1) It has a greater or equal number of dimensions when compared to the
    distribution's event_shape: ``len(values.shape) >= len(dist.event_shape)``
    2) The supplied values' shape is compatible with the distribution's shape
    this means that we check if the righymost ``K`` axes of the values' shape
    match the rightmost ``K`` dimensions of the ``dist_shape``, where ``K`` is
    the minimum between ``len(values.shape)`` and ``len(dist_shape)``.

    Parameters
    ----------
    scoped_name: str
        The variable's scoped name
    values: Any
        The supplied values
    dist: distribution.Distribution
        The ``Distribution`` instance.

    Returns
    -------
    None

    Raises
    ------
    EvaluationError
        When the ``values`` shape is not compatible with the ``Distribution``'s
        shape.
    """
    event_shape = dist.event_shape
    batch_shape = dist.batch_shape
    assert_values_compatible_with_distribution_shape(scoped_name, values, batch_shape, event_shape)


def assert_values_compatible_with_distribution_shape(
    scoped_name: str, values: Any, batch_shape: tf.TensorShape, event_shape: tf.TensorShape
) -> None:
    """Assert if a supplied values are compatible with a distribution's TensorShape.

    A distribution's ``TensorShape``, ``dist_shape``, is made up by the sum of
    the ``batch_shape`` and the ``event_shape``.

    A value is considered to have a consistent shape with the distribution if
    two conditions are met.
    1) It has a greater or equal number of dimensions when compared to the
    distribution's event_shape: ``len(values.shape) >= len(dist.event_shape)``
    2) The supplied values' shape is compatible with the distribution's shape
    this means that we check if the righymost ``K`` axes of the values' shape
    match the rightmost ``K`` dimensions of the ``dist_shape``, where ``K`` is
    the minimum between ``len(values.shape)`` and ``len(dist_shape)``.

    Parameters
    ----------
    scoped_name: str
        The variable's scoped name
    values: Any
        The supplied values
    batch_shape: tf.TensorShape
        The ``tf.TensorShape`` batch_shape instance.
    event_shape: tf.TensorShape
        The ``tf.TensorShape`` event_shape instance.

    Returns
    -------
    None

    Raises
    ------
    EvaluationError
        When the ``values`` shape is not compatible with the ``dist_shape``.
    """
    value_shape = get_observed_tensor_shape(values)
    dist_shape = batch_shape + event_shape
    value_rank = value_shape.rank
    dist_rank = dist_shape.rank
    # TODO: Make the or condition less ugly but at the same time compatible with
    # tf.function. tf.math.maximum makes things kind of weird and raises errors
    if (
        value_rank < event_shape.rank
        or (
            dist_rank < value_rank
            and not dist_shape.is_compatible_with(value_shape[value_rank - dist_rank :])
        )
        or (
            value_rank < dist_rank
            and not value_shape.is_compatible_with(dist_shape[dist_rank - value_rank :])
        )
        or (value_rank == dist_rank and not value_shape.is_compatible_with(dist_shape))
    ):
        raise EvaluationError(
            EvaluationError.INCOMPATIBLE_VALUE_AND_DISTRIBUTION_SHAPE.format(
                scoped_name, dist_shape, value_shape
            )
        )


def get_observed_tensor_shape(arr: Any) -> tf.TensorShape:
    """Extract the supplied arr's shape and return it as a ``tf.TensorShape``.

    Parameters
    ----------
    arr: Any
        Will be tf.convert_to_tensor and the resulting tensor's shape will be
        returned

    Returns
    -------
    output: tf.TensorShape
        The array's shape converted to a ``tf.TensorShape`` instance

    Raises
    ------
    TypeError
        When ``arr`` does not have a ``shape`` attribute.
    """
    return tf.convert_to_tensor(arr).shape
