from typing import Optional, Dict, Any
import tensorflow as tf
import tensorflow_probability as tfp
from pymc4 import flow
from pymc4.coroutine_model import Model
from pymc4.inference.sampling import build_logp_and_deterministic_functions, vectorize_logp_function

tfd = tfp.distributions
tfb = tfp.bijectors

V1_optimizer = tf.python.training.optimizer.Optimizer
V2_optimizer = tf.python.keras.optimizer_v2.optimizer_v2.OptimizerV2
ConvergenceCriterion = tfp.optimizer.convergence_criteria.ConvergenceCriterion


class Approximation(object):
    def __init__(
        self,
        model: Model,
        *,
        optimizer: Optional[V1_optimizer, V2_optimizer] = None,
        convergence_criteria: Optional[ConvergenceCriterion] = None,
        random_seed: int = None,

    ):
        pass


class MeanField(Approximation):
    pass


class FullRank(Approximation):
    pass


def fit(
    model: Model,
    *,
    num_steps: int = 10000,
    method: str = "advi",
    sample_size: int = 1,
    random_seed: int = None,
    observed: Optional[Dict[str, Any]] = None,
    state: Optional[flow.SamplingState] = None,
):
    """
    pass
    """
    (
        logpfn,
        init,
        _deterministics_callback,
        deterministic_names,
        state_,
    ) = build_logp_and_deterministic_functions(
        model, state=state, observed=observed, collect_reduced_log_prob=True,
    )

    if optimizer:
        opt = optimizer
    else:
        opt = tf.optimizers.Adam(learning_rate=0.1)

    loss = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=None,
        surrogate_posterior=None,
        optimizer=opt,
        num_steps=num_steps,
        seed=random_seed,
    )

    return loss
