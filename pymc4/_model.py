import copy
import ast

from . import _template_contexts as contexts
from .ast_compiler import uncompile, parse_snippet, recompile, AutoNameTransformer

import numpy as np
import arviz as az
import tensorflow as tf
import tensorflow_probability as tfp

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

    def make_log_prob_function(self, logp_sum_func=None):
        """Return the log probability of the model."""
        if logp_sum_func is None:
            logp_sum_func = self.logp_sum_single

        def log_prob(*args, logp_sum_func=logp_sum_func, **kwargs):
            varnames = [v.name for v in self._forward_context.vars]
            if args:
                kwargs.update({k:v for k,v in zip(varnames, args)})
            already_specified = set(self._observations.keys()).intersection(set(kwargs.keys()))
            if already_specified:
                raise ValueError('Passed variables {} already specified.'
                                .format(', '.join(already_specified)))
            kwargs.update(self._observations)
            missing_vals = set(varnames) - set(kwargs.keys())
            if missing_vals:
                raise ValueError('Missing value for {} in logp.'
                                .format(', '.join(missing_vals)))
            context = contexts.InferenceContext([kwargs[v.name] for v in self._forward_context.vars], expected_vars=self._forward_context.vars)
            with context:
                self._evaluate()
                logp = logp_sum_func(context)
                return logp

        return log_prob

    def logp_sum_single(self, context):
        return sum(tf.reduce_sum(var.log_prob()) for var in context.vars)

    def logp_sum_multi(self, context):
        return sum(tf.reduce_sum(
            var.log_prob(),
            axis=tf.range(1, tf.rank(var.log_prob()))) for var in context.vars)

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

    def sample_posterior(self, samples=1000, tuning=500, leapfrog_steps=10, step_size=0.005, num_chains=1, compile=True):
        """Samples from the posterior of model."""
        samples += tuning

        if compile:
            function = tf.function
        else:
            function = lambda x: x

        if num_chains == 1:
            forward_sample = self.forward_sample()
            logp_sum_func = self.logp_sum_single
        else:
            forward_sample = tf.vectorized_map(self.forward_sample, tf.range(num_chains))
            logp_sum_func = self.logp_sum_multi

        log_prob_func = function(self.make_log_prob_function(logp_sum_func=logp_sum_func))
        # Create input tensors
        var_names = [var.name for var in self._forward_context.vars if var.name not in self._observations.keys()]


        initial_state = [
            forward_sample[var_name] for var_name in var_names
        ]

        @function
        def hmc_tfp(log_prob_func, initial_state, samples, tuning, leapfrog_steps, step_size):
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob_func,
                num_leapfrog_steps=leapfrog_steps,
                step_size=step_size)

            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=kernel, num_adaptation_steps=int(tuning * 0.8))

            trace, stats = tfp.mcmc.sample_chain(
                num_results=samples,
                num_burnin_steps=tuning,
                current_state=initial_state,
                kernel=kernel,
            )

            return trace, stats

        trace, stats = hmc_tfp(log_prob_func, initial_state, samples, tuning, leapfrog_steps, step_size)
        # Create arviz trace
        # trace_az = {var_name: np.swapaxes(t.numpy()[tuning:], 0, 1) for var_name, t in zip(var_names, trace)}
        trace_az = {var_name: t.numpy()[tuning:][np.newaxis, ...] for var_name, t in zip(var_names, trace)}
        trace_az = az.from_dict(trace_az)
        return trace_az, stats
