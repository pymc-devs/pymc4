import copy
from . import _template_contexts as contexts

import tensorflow as tf


__all__ = ['model']


def model(func):
    return ModelTemplate(func)


class ModelTemplate:
    def __init__(self, func):
        self._func = func

    def configure(self, *args, **kwargs):
        with contexts.ForwardContext() as context:
            self._func(*args, **kwargs)
        return Model(self, context, template_args=(args, kwargs))


class Model:
    def __init__(self, template, forward_context, template_args):
        self._template = template
        self._forward_context = forward_context
        self._observations = {}

    def make_logp_function(self):
        def logp(*args):
            context = contexts.InferenceContext(
                args, expected_vars=self._forward_context.vars)
            with context:
                self._template._func()

                var_logps = []
                for var in context.vars:
                    var_logps.append(var._distribution.log_prob(var))
                return sum(tf.reduce_sum(val) for val in var_logps)

        return logp

    def observe(self, **kwargs):
        model = copy.copy(self)
        model.observations.update(kwargs)

