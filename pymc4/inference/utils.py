from typing import Optional, Tuple, List
import numpy as np
import arviz as az

from .. import Model, flow


def initialize_sampling_state(
    model: Model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None
) -> Tuple[flow.SamplingState, List[str]]:
    """
    Initilize the model provided state and/or observed variables.

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
    _, state = flow.evaluate_model_transformed(model, observed=observed, state=state)
    deterministic_names = list(state.deterministics)

    state, transform_after = state.as_sampling_state()
    return state, deterministic_names + transform_after


def trace_to_arviz(pm4_trace, pm4_sample_stats):
    """
    Tensorflow to Arviz trace convertor.

    Convert a PyMC4 trace as returned by sample() to an ArviZ trace object
    that can be passed to e.g. arviz.plot_trace().

    Parameters
    ----------
    pm4_trace : dict
    pm4_sample_stats : dict

    Returns
    -------
    arviz.data.inference_data.InferenceData
    """

    posterior = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in pm4_trace.items() if "/" in k}
    sample_stats = {k: v.numpy().T for k, v in pm4_sample_stats.items()}
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)
