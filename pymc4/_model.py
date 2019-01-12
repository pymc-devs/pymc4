import copy
from . import _template_contexts as contexts

import tensorflow as tf


__all__ = ["model"]


def model(func):
    return ModelTemplate(func)


class ModelTemplate:
    def __init__(self, func):
        self._func = func

    def configure(self, *args, **kwargs):
        with contexts.ForwardContext() as context:
            model = Model(self, context, template_args=(args, kwargs))
            model._evaluate()
        return model


class Model:
    def __init__(self, template, forward_context, template_args):
        self._template = template
        self._template_args = template_args
        self._forward_context = forward_context
        self._observations = {}

    def _evaluate(self):
        """Call the template function with the saved arguments."""
        args, kwargs = self._template_args
        self._template._func(*args, **kwargs)
        return

    def make_log_prob_function(self):
        def log_prob(*args):
            context = contexts.InferenceContext(args, expected_vars=self._forward_context.vars)
            with context:
                self._evaluate()
                return sum(tf.reduce_sum(var.log_prob()) for var in context.vars)

        return log_prob

    def forward_sample(self, *args, **kwargs):
        with self._forward_context as context:
            samples = {var.name: var.as_tensor() for var in context.vars}
        return samples

    def observe(self, **kwargs):
        model = copy.copy(self)
        model._observations.update(kwargs)
        return model
