from typing import Optional

from pymc4 import Model, flow


def initialize_state(model: Model, observed: Optional[dict] = None) -> flow.SamplingState:
    """
    Initilize the model provided state and/or observed variables

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
