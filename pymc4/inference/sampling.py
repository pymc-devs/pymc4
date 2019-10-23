from typing import Optional
import tensorflow as tf
from tensorflow_probability import mcmc
from pymc4.inference.utils import initialize_state
from pymc4.coroutine_model import Model
from pymc4 import flow


def sample(
    model: Model,
    num_samples=1000,
    num_chains=10,
    burn_in=100,
    step_size=0.1,
    observed: Optional[dict] = None,
    state: Optional[flow.SamplingState] = None,
    nuts_kwargs=None,
    adaptation_kwargs=None,
    sample_chain_kwargs=None,
    xla=False,
):
    """
    Perform MCMC sampling using NUTS (for now).

    Parameters
    ----------
    model : pymc4.Model
        Model to sample posterior for
    num_samples : int
        Num samples in a chain
    num_chains : int
        Num chains to run
    burn_in : int
        Length of burn-in period
    step_size : float
        Initial step size
    observed : Optional[dict]
        New observed values (optional)
    state : Optional[pymc4.flow.SamplingState]
        Alternative way to pass specify initial values and observed values
    nuts_kwargs : Optional[dict]
        Pass non-default values for nuts kernel, see
        ``tensorflow_probability.experimental.mcmc.NoUTurnSamplerUnrolled`` for options
    adaptation_kwargs : Optional[dict]
        Pass non-default values for nuts kernel, see
        ``tensorflow_probability.mcmc.dual_averaging_step_size_adaptation.DualAveragingStepSizeAdaptation`` for options
    sample_chain_kwargs : dict
        Pass non-default values for nuts kernel, see
        ``tensorflow_probability.mcmc.sample_chain`` for options
    xla : bool
        Enable experimental XLA

    Returns
    -------
    Trace

    Examples
    --------
    Let's start with a simple model. We'll need some imports to experiment with it.

    >>> import pymc4 as pm
    >>> from pymc4 import distributions as dist
    >>> import numpy as np

    This particular model has a latent variable `sd`

    >>> @pm.model
    ... def nested_model(cond):
    ...     sd = yield dist.HalfNormal("sd", 1., transform=dist.transforms.Log())  #TODO: Auto-transform
    ...     norm = yield dist.Normal("n", cond, sd, observed=np.random.randn(10))
    ...     return norm

    Now, we may want to perform sampling from this model. We already observed some variables and we now need to fix
    the condition.

    >>> conditioned = nested_model(cond=2.)

    Passing ``cond=2.`` we condition our model for future evaluation. Now we go to sampling. Nothing special is required
    but passing the model to ``pm.sample``, the rest configuration is held by PyMC4.

    >>> trace = sample(conditioned)

    Notes
    -----
    Things that are considered to be under discussion are overriding observed variables. The API for that may look like

    >>> new_observed = {"nested_model/n": np.random.randn(10) + 1}
    >>> trace = sample(conditioned, observed=new_observed)

    This will give a trace with new observed variables. This way is considered to be explicit.

    """
    wrapper = LogProbDeterministicWrapper(model, state=state, observed=observed)
    parallel_logpfn = wrapper.logpfn()
    init = wrapper.state
    init_state = list(init.values())
    init_keys = list(init.keys())
    init_state = tile_init(init_state, num_chains)

    def trace_fn(_, pkr):
        return (
            pkr.inner_results.target_log_prob,
            pkr.inner_results.leapfrogs_taken,
            pkr.inner_results.has_divergence,
            pkr.inner_results.energy,
            pkr.inner_results.log_accept_ratio,
        )

    @tf.function(autograph=False)
    def run_chains(init, step_size):
        nuts_kernel = mcmc.NoUTurnSampler(
            target_log_prob_fn=parallel_logpfn, step_size=step_size, **(nuts_kwargs or dict())
        )
        adapt_nuts_kernel = mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=burn_in,
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            **(adaptation_kwargs or dict()),
        )

        results, sample_stats = mcmc.sample_chain(
            num_samples,
            current_state=init,
            kernel=adapt_nuts_kernel,
            num_burnin_steps=burn_in,
            trace_fn=trace_fn,
            **(sample_chain_kwargs or dict()),
        )

        return results, sample_stats

    if xla:
        results, sample_stats = tf.xla.experimental.compile(
            run_chains, inputs=[init_state, step_size]
        )
    else:
        results, sample_stats = run_chains(init_state, step_size)

    posterior = dict(zip(init_keys, results))
    # Keep in sync with pymc3 naming convention
    sampler_stats = dict(
        zip(["lp", "tree_size", "diverging", "energy", "mean_tree_accept"], sample_stats)
    )
    return posterior, sampler_stats


def build_logp_function(
    model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None
):
    if not isinstance(model, Model):
        raise TypeError(
            "`sample` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                type(model)
            )
        )
    if state is not None and observed is not None:
        raise ValueError("Can't use both `state` and `observed` arguments")
    if state is None:
        state = initialize_state(model, observed=observed)
    else:
        state = state.as_sampling_state()

    observed = state.observed_values
    unobserved_keys, unobserved_values = zip(*state.all_unobserved_values.items())

    @tf.function(autograph=False)
    def logpfn(*values, **kwargs):
        if kwargs and values:
            raise TypeError("Either list state should be passed or a dict one")
        elif values:
            kwargs = dict(zip(unobserved_keys, values))
        st = flow.SamplingState.from_values(kwargs, observed_values=observed)
        _, st = flow.evaluate_model_transformed(model, state=st)
        return st.collect_log_prob_and_deterministic()

    return logpfn, dict(state.all_unobserved_values)


def vectorize_logp_function(logpfn):
    # TODO: vectorize with dict
    def vectorized_logpfn_and_deterministics(*state):
        return tf.nest.map_structure(tf.vectorized_map, logpfn(*state))
#    def vectorized_logpfn(*state):
#        return tuple([
#            tf.vectorized_map(lambda mini_state: logpfn(*mini_state), state)
#        ])

    return vectorized_logpfn_and_deterministics


class LogProbDeterministicWrapper:
    __slots__ = ("_func", "_state", "deterministic_dists", "deterministic_values")

    def __init__(self, model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None):
        logpfn, self._state = build_logp_function(model=model, observed=observed, state=state)
        self._func = vectorize_logp_function(logpfn)
        # TODO: Use model state instead of self._state because part of it can be overwritten
        self.deterministic_dists = self.state.deterministics

    def __call__(self, *args, **kwargs):
        output = self._func(*args, **kwargs)
        log_prob = output[0]
        self.deterministic_values = output[1:]
        return log_prob

    @property
    def logpfn(self):
        return self.__call__

    @property
    def state(self):
        return self._state

    def get_deterministic_values(self, *):
        return dict(zip(self.deterministic_dists, self.deterministic_values))


def tile_init(init, num_repeats):
    return [tf.tile(tf.expand_dims(tens, 0), [num_repeats] + [1] * tens.ndim) for tens in init]
