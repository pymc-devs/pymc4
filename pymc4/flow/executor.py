import collections
import pymc4 as pm
import types
import abc
from pymc4 import scopes
from .. import utils
from pymc4.distributions import abstract


class EvaluationError(RuntimeError):
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


class Executor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def new_state(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def proceed_distribution(self, dist, model_info, state):
        raise NotImplementedError

    @abc.abstractmethod
    def modify_distribution(self, dist, model_info, state):
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_model_control_flow(self, model, model_info, state):
        raise NotImplementedError

    @abc.abstractmethod
    def finalize_control_flow(self, stop_iteration, model_info, state):
        raise NotImplementedError

    def evaluate_model(self, model, *args, state=None, **kwargs):
        if state is None:
            state = self.new_state(*args, **kwargs)
        else:
            if args or kwargs:
                raise ValueError("Provided arguments along with not empty state")
        if isinstance(model, pm.coroutine_model.Model):
            model_info = model.model_info()
            try:
                control_flow = self.prepare_model_control_flow(model, model_info, state)
            except EarlyReturn as e:
                return e.args
        else:
            if not isinstance(model, types.GeneratorType):
                raise StopExecution(
                    "Attempting to call `evaluate_model` on a "
                    "non model-like object {}. Supported types are "
                    "`types.GeneratorType` and `pm.coroutine_model.Model`"
                )
            control_flow = model
            model_info = pm.coroutine_model.Model.default_model_info()
        return_value = None
        while True:
            try:
                with model_info["scope"]:
                    dist = control_flow.send(return_value)
                    if isinstance(dist, abstract.Distribution):
                        dist = self.modify_distribution(dist, model_info, state)
                    if dist is None:
                        return_value = None
                    elif isinstance(dist, abstract.Distribution):
                        try:
                            return_value, state = self.proceed_distribution(dist, model_info, state)
                        except EvaluationError as error:
                            control_flow.throw(error)
                            raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
                    elif isinstance(dist, (pm.coroutine_model.Model, types.GeneratorType)):
                        return_value, state = self.evaluate_model(dist, state=state)
                    else:
                        error = EvaluationError(
                            "Type of {} can't be processed in evaluation".format(dist)
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
                return_value, state = self.finalize_control_flow(stop_iteration, model_info, state)
                break
        return return_value, state

    __call__ = evaluate_model


SamplingState = collections.namedtuple("EvaluationState", "values,distributions,potentials")


class SamplingExecutor(Executor):
    def __init__(self, *conditions: dict, **condition_kwargs: dict):
        self.default_condition = utils.merge_dicts(*conditions, condition_kwargs)

    def new_state(self, *conditions: dict, **condition_kwargs: dict):
        condition_state = self.default_condition.copy()
        condition_state.update(utils.merge_dicts(*conditions, condition_kwargs))
        return SamplingState(values=condition_state, distributions=dict(), potentials=[])

    def modify_distribution(self, dist, model_info, state):
        return dist

    def proceed_distribution(self, dist, model_info, state):
        if isinstance(dist, abstract.Potential):
            value = dist.value
            state.potentials.append(dist)
            return value, state

        scoped_name = scopes.variable_name(dist.name)
        if scoped_name in state.distributions:
            raise EvaluationError(
                "Attempting to create duplicate variable '{}', "
                "this may happen if you forget to use `pm.name_scope()` when calling same "
                "model/function twice without providing explicit names. If you see this "
                "error message and the function being called is not wrapped with "
                "`pm.model`, you should better wrap it to provide explicit name for this model".format(
                    scoped_name
                )
            )
        if scoped_name in state.values:
            return_value = state.values[scoped_name]
        else:
            return_value = state.values[scoped_name] = dist.sample()
        state.distributions[scoped_name] = dist
        return return_value, state

    def prepare_model_control_flow(self, model, model_info, state):
        control_flow: types.GeneratorType = model.control_flow()
        model_name = model_info["name"]
        if model_name is None and model_info["keep_return"]:
            error = EvaluationError("Attempting to create unnamed variable")
            control_flow.throw(error)
            control_flow.close()
            raise StopExecution(StopExecution.NOT_HELD_ERROR_MESSAGE) from error
        return_name = scopes.variable_name(model_name)
        if not model_info["keep_auxiliary"] and return_name in state.values:
            raise EarlyReturn(state.values[model_name], state)
        return control_flow

    def finalize_control_flow(self, stop_iteration, model_info, state):
        if stop_iteration.args:
            return_value = stop_iteration.args[0]
        else:
            return_value = None
        if return_value is not None and model_info["keep_return"]:
            return_name = scopes.variable_name(model_info["name"])
            state.values[return_name] = return_value
        return return_value, state
