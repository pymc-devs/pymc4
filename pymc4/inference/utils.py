from typing import Optional
import numpy as np

from .. import Model, flow


def initialize_state(model: Model, observed: Optional[dict] = None) -> flow.SamplingState:
    """
    Initilize the model provided state and/or observed variables.

    Parameters
    ----------
    model : pymc4.Model
    observed : Optional[dict]

    Returns
    -------
    pymc4.flow.SamplingState
    """
    _, state = flow.evaluate_model_transformed(model, observed=observed)
    return state.as_sampling_state()


def trace_to_arviz(pm4_trace):
    """
    Tensorflow to Arviz trace convertor.

    Convert a PyMC4 trace as returned by sample() to an ArviZ trace object
    that can be passed to e.g. arviz.plot_trace().

    Parameters
    ----------
    pm4_trace : dict

    Returns
    -------
    arviz.data.inference_data.InferenceData
    """
    import arviz as az

    az_dict = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in pm4_trace.items()}
    return az.from_dict(az_dict)
