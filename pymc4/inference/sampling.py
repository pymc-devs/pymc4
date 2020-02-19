from typing import Optional, Dict, Any
import tensorflow as tf
from pymc4.coroutine_model import Model
from pymc4 import flow
from pymc4.mcmc.samplers import reg_samplers


def sample(
    model: Model,
    sampler: str = "nuts",  # TODO: to keep current progress, later, assigner should be added
    num_samples: int = 1000,
    num_chains: int = 10,
    burn_in: int = 100,
    step_size: float = 0.1,
    observed: Optional[Dict[str, Any]] = None,
    state: Optional[flow.SamplingState] = None,
    xla: bool = False,
    use_auto_batching: bool = True,
    **kwargs,
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
    observed : Optional[Dict[str, Any]]
        New observed values (optional)
    state : Optional[pymc4.flow.SamplingState]
        Alternative way to pass specify initial values and observed values
    xla : bool
        Enable experimental XLA
    use_auto_batching : bool
        WARNING: This is an advanced user feature. If you are not sure how to use this, please use
        the default ``True`` value.
        If ``True``, the model's total ``log_prob`` will be automatically vectorized to work across
        multiple indepedent chains using ``tf.vectorized_map``. If ``False``, the model is assumed
        be defined in vectorized way. This means that every distribution has the proper
        ``batch_shape`` and ``event_shape``s so that all the outputs from each distribution's
        ``log_prob`` will broadcast with each other, and that the forward passes through the model
        (prior and posterior predictive sampling) all work on values with any value of
        ``batch_shape``. Achieving this is a hard task, but it enables the model to be safely
        evaluated in parallel across all chains in MCMC, so sampling will be faster than in the
        automatically batched scenario.

    Returns
    -------
    Trace : InferenceDataType
        An ArviZ's InferenceData object with the groups: posterior, sample_stats and observed_data

    Examples
    --------
    Let's start with a simple model. We'll need some imports to experiment with it.

    >>> import pymc4 as pm
    >>> import numpy as np

    This particular model has a latent variable `sd`

    >>> @pm.model
    ... def nested_model(cond):
    ...     sd = yield pm.HalfNormal("sd", 1.)
    ...     norm = yield pm.Normal("n", cond, sd, observed=np.random.randn(10))
    ...     return norm

    Now, we may want to perform sampling from this model. We already observed some variables and we
    now need to fix the condition.

    >>> conditioned = nested_model(cond=2.)

    Passing ``cond=2.`` we condition our model for future evaluation. Now we go to sampling.
    Nothing special is required but passing the model to ``pm.sample``, the rest configuration is
    held by PyMC4.

    >>> trace = sample(conditioned)

    Notes
    -----
    Things that are considered to be under discussion are overriding observed variables. The API
    for that may look like

    >>> new_observed = {"nested_model/n": np.random.randn(10) + 1}
    >>> trace = sample(conditioned, observed=new_observed)

    This will give a trace with new observed variables. This way is considered to be explicit.

    """
    try:
        sampler = reg_samplers[sampler]
    except KeyError:
        print("The given sampler doesn't exist")
    sampler = sampler(model, **kwargs, step_size=step_size, num_adaptation_steps=burn_in)
    return sampler(
        num_samples=num_samples,
        num_chains=num_chains,
        burn_in=burn_in,
        observed=observed,
        state=state,
        use_auto_batching=use_auto_batching,
        xla=xla,
    )
