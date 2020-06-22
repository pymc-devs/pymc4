import tensorflow as tf
from typing import Optional, Tuple, List
import numpy as np
import arviz as az

from pymc4 import Model, flow


def initialize_sampling_state(
    model: Model,
    observed: Optional[dict] = None,
    state: Optional[flow.SamplingState] = None,
    num_chains: Optional[int] = 1,
    draws: Optional[int] = 1,
    is_smc: Optional[bool] = False,
    smc_run: Optional[bool] = False,
) -> Tuple[flow.SamplingState, List[str]]:
    """
    Initialize the model provided state and/or observed variables.

    Parameters
    ----------
    model : pymc4.Model
    observed : Optional[dict]
    state : Optional[flow.SamplingState]

    Returns
    -------
    state: pymc4.flow.SamplingState
        The model's sampling state
    deterministic_names: List[str]
        The list of names of the model's deterministics
    """
    eval_func = flow.evaluate_model_transformed
    if is_smc is False:
        eval_func = flow.evaluate_meta_model
    _, state = eval_func(
        model, observed=observed, state=state, num_chains=num_chains, draws=draws, smc_run=smc_run
    )
    deterministic_names = list(state.deterministics)

    state, transformed_names = state.as_sampling_state(smc_run=smc_run)
    return state, deterministic_names + transformed_names


def trace_to_arviz(
    trace=None,
    sample_stats=None,
    observed_data=None,
    prior_predictive=None,
    posterior_predictive=None,
    inplace=True,
):
    """
    Tensorflow to Arviz trace convertor.

    Creates an ArviZ's InferenceData object with inference, prediction and/or sampling data
    generated by PyMC4

    Parameters
    ----------
    trace : dict or InferenceData
    sample_stats : dict
    observed_data : dict
    prior_predictive : dict
    posterior_predictive : dict
    inplace : bool

    Returns
    -------
    ArviZ's InferenceData object
    """
    if trace is not None and isinstance(trace, dict):
        trace = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in trace.items() if "/" in k}
    if sample_stats is not None and isinstance(sample_stats, dict):
        sample_stats = {k: v.numpy().T for k, v in sample_stats.items()}
    if prior_predictive is not None and isinstance(prior_predictive, dict):
        prior_predictive = {k: v[np.newaxis] for k, v in prior_predictive.items()}
    if posterior_predictive is not None and isinstance(posterior_predictive, dict):
        if isinstance(trace, az.InferenceData) and inplace == True:
            return trace + az.from_dict(posterior_predictive=posterior_predictive)
        else:
            trace = None

    return az.from_dict(
        posterior=trace,
        sample_stats=sample_stats,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
    )


def vectorize_logp_function(logpfn):
    # TODO: vectorize with dict
    def vectorized_logpfn(*state):
        return tf.vectorized_map(lambda mini_state: logpfn(*mini_state), state)

    return vectorized_logpfn


def tile_init(init, num_repeats):
    return [tf.tile(tf.expand_dims(tens, 0), [num_repeats] + [1] * tens.ndim) for tens in init]
