from typing import Optional, Dict, Any
import tensorflow as tf
import tensorflow_probability as tfp
from pymc4.coroutine_model import Model
from pymc4 import flow
from pymc4.inference.utils import initialize_sampling_state, vectorize_logp_function

from tensorflow_probability.python.experimental.mcmc.sample_sequential_monte_carlo import (
    make_rwmh_kernel_fn,
)

sample_sequential_monte_carlo_chain = tfp.experimental.mcmc.sample_sequential_monte_carlo


def sample_smc(
    model: Model,
    draws: int = 5000,
    num_chains: int = 10,
    state: Optional[flow.SamplingState] = None,
    observed: Optional[Dict[str, Any]] = None,
    xla: bool = False,
):
    """
    Perform SMC
    TODO: Add docs
    """
    (logpfn_prior, logpfn_lkh, init, state_,) = _build_logp_smc(
        model, num_chains=num_chains, draws=draws, state=state, observed=observed,
    )
    for k in init.keys():
        init[k] = state_.all_unobserved_values_batched[k]
    init_state = list(init.values())

    parallel_logpfn_prior = logpfn_prior
    parallel_logpfn_lkh = logpfn_lkh

    @tf.function(autograph=False)
    def run_smc(init):
        n_stage, final_state, final_kernel_results = sample_sequential_monte_carlo_chain(
            parallel_logpfn_prior,
            parallel_logpfn_lkh,
            init,
            make_kernel_fn=make_rwmh_kernel_fn,
            max_num_steps=50,
        )
        return n_stage, final_state, final_kernel_results

    if xla:
        _, final_state, _ = tf.xla.experimental.compile(run_smc, inputs=[init_state])
    else:
        _, final_state, _ = run_smc(init_state)
    return final_state


def _build_logp_smc(
    model,
    num_chains,
    draws,
    observed: Optional[dict] = None,
    state: Optional[flow.SamplingState] = None,
):
    # TODO: modified, merged with the sampling implementation
    # there is a better logic in more samplers implementation
    # leave for now
    if not isinstance(model, Model):
        raise TypeError(
            "`sample` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                type(model)
            )
        )
    if state is not None and observed is not None:
        raise ValueError("Can't use both `state` and `observed` arguments")

    state, _ = initialize_sampling_state(
        model,
        observed=observed,
        state=state,
        num_chains=num_chains,
        draws=draws,
        is_smc=True,
        smc_run=True,
    )

    if not state.all_unobserved_values:
        raise ValueError(
            f"Can not calculate a log probability: the model {model.name or ''} has no unobserved values."
        )

    unobserved_keys, unobserved_values = zip(*state.all_unobserved_values.items())

    @tf.function(autograph=False)
    def logpfn_likelihood(*values, **kwargs):
        if kwargs and values:
            raise TypeError("Either list state should be passed or a dict one")
        elif values:
            kwargs = dict(zip(unobserved_keys, values))
        st = flow.SamplingState.from_values(kwargs, observed_values=observed)
        _, st = flow.evaluate_model_transformed(
            model, state=st, num_chains=num_chains, draws=draws, smc_run=True
        )
        return st.collect_log_prob_smc(is_prior=False)

    @tf.function(autograph=False)
    def logpfn_prior(*values, **kwargs):
        if kwargs and values:
            raise TypeError("Either list state should be passed or a dict one")
        elif values:
            kwargs = dict(zip(unobserved_keys, values))
        st = flow.SamplingState.from_values(kwargs, observed_values=observed)
        _, st = flow.evaluate_model_transformed(
            model, state=st, num_chains=num_chains, draws=draws, smc_run=True
        )
        return st.collect_log_prob_smc(is_prior=True)

    return (
        logpfn_prior,
        logpfn_likelihood,
        dict(state.all_unobserved_values),
        state,
    )
