from typing import Optional, Tuple, List
import numpy as np
import arviz as az

from pymc4 import Model, flow


def initialize_sampling_state(
    model: Model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None
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
    _, state = flow.evaluate_meta_model(model, observed=observed, state=state)
    deterministic_names = list(state.deterministics)

    state, transformed_names = state.as_sampling_state()
    return state, deterministic_names + transformed_names


def initialize_state(model: Model) -> Tuple[List[str]]:
    """
    Get list of discrete/continuous distributions

    Parameters
    ----------
    model : pymc4.Model

    Returns
    -------
    discrete_names: List[str]
        The list of discrete distributions
    continuous_names: List[str]
        The list of continuous distributions
    """
    _, state = flow.evaluate_model_transformed(model)
    discrete_names, continuous_names = (
        list(state.discrete_distributions),
        list(state.continuous_distributions),
    )
    return state


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
