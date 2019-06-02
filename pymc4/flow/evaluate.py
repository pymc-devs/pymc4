import collections
import pymc4 as pm
import types

import pymc4.scopes

EvaluationState = collections.namedtuple("EvaluationState", "values,distributions")


class EvaluationError(RuntimeError):
    ...


class StopExecution(StopIteration):
    MESSAGE = """
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


def evaluate_model(modelgen, state=None):
    if state is None:
        state = EvaluationState(
            values=dict(),
            distributions=dict(),
        )

    if isinstance(modelgen, pm.coroutine_model.Model):
        model_name = modelgen.name
        gen: types.GeneratorType = iter(modelgen)
        if model_name is None and modelgen.keep_return:
            error = EvaluationError("Attempting to create unnamed variable")
            gen.throw(error)
            raise StopExecution(StopExecution.MESSAGE) from error
        with pymc4.scopes.name_scope(model_name):
            return_name = pymc4.scopes.Scope.variable_name(modelgen.name)
            keep_return = modelgen.keep_return
        if not modelgen.keep_auxiliary and return_name in state.values:
            return state.values[model_name], state
    else:
        gen = modelgen
        keep_return = False
        return_name = None
    sample = None
    while True:
        try:
            dist = gen.send(sample)
            if isinstance(dist, pm.distributions.Distribution):
                name = pymc4.scopes.Scope.variable_name(dist.name)
                if name in state.distributions:
                    error = EvaluationError(
                        "Attempting to create duplicate variable '{}', "
                        "this may happen if you forget to use `pm.name_scope()` when calling same "
                        "model/function twice without providing explicit names. If you see this "
                        "error message and the function being called is not wrapped with "
                        "`pm.model`, you should better wrap it to provide explicit name for this model"
                        .format(name))
                    gen.throw(error)
                    raise StopExecution(error)
                if name in state.values:
                    sample = state.values[name]
                else:
                    sample = state.values[name] = dist.sample()
                state.distributions[name] = dist
            elif isinstance(dist, (pm.coroutine_model.Model, types.GeneratorType)):
                sample, state = evaluate_model(dist, state=state)
            else:
                error = EvaluationError("Type of {} can't be processed in evaluation".format(dist))
                gen.throw(error)
                raise StopExecution(StopExecution.MESSAGE) from error
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
            raise
        except StopIteration as e:
            if e.args:
                sample = e.args[0]
            else:
                sample = None
            if sample is not None and keep_return:
                state.values[return_name] = sample
            break
    return sample, state
