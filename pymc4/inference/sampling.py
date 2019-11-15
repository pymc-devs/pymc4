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
    logpfn = LogProbDeterministicWrapper(model, state=state, observed=observed)
    init = logpfn.all_unobserved
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
        ) + logpfn.deterministic_values

    @tf.function(autograph=False)
    def run_chains(init, step_size):
        nuts_kernel = mcmc.NoUTurnSampler(
            target_log_prob_fn=logpfn, step_size=step_size, **(nuts_kwargs or dict())
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
    deterministic_names = logpfn.deterministic_names
    sampler_stats = dict(
        zip(
            ["lp", "tree_size", "diverging", "energy", "mean_tree_accept"] + deterministic_names,
            sample_stats,
        )
    )
    return posterior, sampler_stats


class LogProbDeterministicWrapper:
    __slots__ = ("_func", "_state", "all_unobserved", "deterministic_names", "deterministic_values")

    def __init__(
        self, model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None
    ):
        state = self.prepare_state(model=model, observed=observed, state=state)
        
        logp_det_fn, self.all_unobserved = self.build_logp_and_deterministics_function(model=model, state=state)
        self._func = self.vectorize_logp_det_function(logp_det_fn)

    def __call__(self, *args, **kwargs):
        output = self._func(*args, **kwargs)
        log_prob = output[0]
        self.deterministic_values = output[1:]
        return log_prob

    @property
    def state(self):
        return self._state

    def prepare_state(self, model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None):
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
        self._state = state
        self.deterministic_names = list(state.deterministics)
        return state

    def build_logp_and_deterministics_function(
        self, model, state
    ):
        observed = state.observed_values
        deterministics = state.deterministics
        unobserved_keys, unobserved_values = zip(*state.all_unobserved_values.items())

        @tf.function(autograph=False)
        def logp_det_fn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs, observed_values=observed, deterministics=deterministics)
            _, st = flow.evaluate_model_transformed(model, state=st)
            return st.collect_log_prob_and_deterministics()
    
        return logp_det_fn, dict(state.all_unobserved_values)

    def vectorize_logp_det_function(self, logp_det_fn):
        def parallel_logp_det_function(*state):
             return tf.vectorized_map(lambda mini_state: logp_det_fn(*mini_state), state)
        return parallel_logp_det_function


def tile_init(init, num_repeats):
    return [tf.tile(tf.expand_dims(tens, 0), [num_repeats] + [1] * tens.ndim) for tens in init]
