import collections
import pymc4 as pm
import types

import pymc4.scopes

EvaluationState = collections.namedtuple("EvaluationState", "values,priors")


def evaluate_model(modelgen, state=None):
    if state is None:
        state = EvaluationState(
            values=dict(),
            priors=dict(),
        )

    if isinstance(modelgen, pm.coroutine_model.Model):
        model_name = modelgen.name
        gen: types.GeneratorType = iter(modelgen)
        if model_name is None and modelgen.keep_return:
            gen.throw(RuntimeError("Attempting to create unnamed variable"))
            raise StopIteration
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
            try:
                # first, check whether we already dealing with a Distribution
                # if yes, we should sample and record it
                #
                # Conversion function should do most of the stuff,
                # but it will not convert generators and Model objects
                # they will be proceeded by hand
                dist = pm.distributions.convert_distribution(dist)
                name = pymc4.scopes.Scope.variable_name(dist.name)
                if name in state.priors:
                    gen.throw(RuntimeError(
                        "Attempting to create duplicate variable '{}', "
                        "this may happen if you forget to use `pm.name_scope()` when calling same "
                        "model/function twice without providing explicit names. If you see this "
                        "error message and the function being called is not wrapped with "
                        "`pm.model`, you should better wrap it to provide explicit name for this model"
                        .format(name))
                    )
                if name in state.values:
                    sample = state.values[name]
                else:
                    sample = state.values[name] = dist.sample()
                state.priors[name] = dist
            except TypeError as conversion_error:
                # save this error state, as dist may appear not to be the type we support
                if isinstance(dist, (pm.coroutine_model.Model, types.GeneratorType)):
                    sample, state = evaluate_model(dist, state=state)
                else:
                    gen.throw(conversion_error)
        except StopIteration as e:
            if e.args:
                sample = e.args[0]
            else:
                sample = None
            if sample is not None and keep_return:
                state.values[return_name] = sample
            break
    return sample, state
