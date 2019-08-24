from typing import Optional
import tensorflow as tf
from pymc4.inference.utils import initialize_state
from pymc4.coroutine_model import Model
from pymc4 import flow


def sample(
    model: Model, observed: Optional[dict] = None, state: Optional[flow.SamplingState] = None
):
    """
    The main API to perform MCMC sampling using NUTS (for now)

    Parameters
    ----------
    model : pymc4.Model
    observed : Optional[dict]
    state : Optional[pymc4.flow.SamplingState]

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

    >>> trace = sample(model)

    Notes
    -----
    Things that are considered to be under discussion are overriding observed variables. The API for that may look like

    >>> new_observed = {"nested_model/n": np.random.randn(10) + 1}
    >>> trace = sample(model, observed=new_observed)

    This will give a trace with new observed variables. This way is considered to be explicit.

    """
    logpfn, init = build_logp_function(model, state=state, observed=observed)


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
    def logpfn(values):
        st = flow.SamplingState.from_values(
            dict(zip(unobserved_keys, values)), observed_values=observed
        )
        _, st = flow.evaluate_model_transformed(model, state=st)
        return st.collect_log_prob()

    return logpfn, list(unobserved_values)
