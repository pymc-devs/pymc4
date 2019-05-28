import copy
import ast

from . import _template_contexts as contexts
from .ast_compiler import uncompile, parse_snippet, recompile, AutoNameTransformer

import tensorflow as tf


__all__ = ["model"]


def model(_func=None, *, auto_name=False):
    """
    Decorate a model-specification function as a PyMC4 model.

    Parameters
    ----------
    auto_name : bool
        Whether to automatically infer names of RVs.

    Returns
    -------
    The function wrapped in a ModelTemplate object.
    """

    def wrap(func):
        if auto_name:
            # uncompile function
            unc = uncompile(func.__code__)

            # convert to ast and apply visitor
            tree = parse_snippet(*unc)
            AutoNameTransformer().visit(tree)
            ast.fix_missing_locations(tree)
            unc[0] = tree

            # recompile and patch function's code
            func.__code__ = recompile(*unc)

        return ModelTemplate(func)

    if _func is None:
        return wrap
    else:
        return wrap(_func)


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

        def log_prob(**kwargs):
            vars = self._forward_context.vars
            if set(self._observations.keys()).intersection(set(kwargs.keys())):
                raise ValueError('Passed variable already specified.')
            kwargs.update(self._observations)
            if len(kwargs) != len(vars):
                raise ValueError('Missing value for variable in logp.')
            context = contexts.InferenceContext([kwargs[v.name] for v in self._forward_context.vars], expected_vars=self._forward_context.vars)
            with context:
                self._evaluate()
                return sum(tf.reduce_sum(var.log_prob()) for var in context.vars)

        return log_prob

    def forward_sample(self, *args, **kwargs):
        """Simulate data from the model via forward sampling."""
        with self._forward_context as context:
            samples = {var.name: var.sample() for var in context.vars}
        return samples

    def observe(self, **kwargs):
        """Condition the model on observed data."""
        model = copy.copy(self)
        model._observations.update(kwargs)
        return model
