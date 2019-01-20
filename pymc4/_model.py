import copy
from . import _template_contexts as contexts

import tensorflow as tf


__all__ = ["model"]


def model(func):
    """
    Decorate a model-specification function as a PyMC4 model.

    Parameters
    ----------
    func : a function
        The function that specifies the PyMC4 model

    Returns
    -------
    The function wrapped in a ModelTemplate object.
    """
    return ModelTemplate(func)


class ModelTemplate:
    """
    Wrapper object that sets up the model for use later.

    This is an infrastructural piece that end-users are generally not expected
    to be using.
    """

    def __init__(self, func):
        self._func = func

    def configure(self, *args, **kwargs):
        """Configure the model by setting it up as a Model object."""
        with contexts.ForwardContext() as context:
            model = Model(self, context, template_args=(args, kwargs))
            model._evaluate()
        return model


class Model:
    """Base model object."""

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
        """Return the log probability of the model."""

        def log_prob(*args):
            context = contexts.InferenceContext(args, expected_vars=self._forward_context.vars)
            with context:
                self._evaluate()
                return sum(tf.reduce_sum(var.log_prob()) for var in context.vars)

        return log_prob

    def forward_sample(self, *args, **kwargs):
        """Simulate data from the model via forward sampling."""
        with self._forward_context as context:
            samples = {var.name: var.as_tensor() for var in context.vars}
        return samples

    def observe(self, **kwargs):
        """Condition the model on observed data."""
        model = copy.copy(self)
        model._observations.update(kwargs)
        return model
