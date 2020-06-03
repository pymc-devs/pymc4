from typing import Optional
import tensorflow as tf
import tensorflow_probability as tfp
from pymc4 import flow
from pymc4.coroutine_model import Model
from pymc4.inference.utils import initialize_sampling_state

tfd = tfp.distributions
tfb = tfp.bijectors

V1_optimizer = tf.python.training.optimizer.Optimizer
V2_optimizer = tf.python.keras.optimizer_v2.optimizer_v2.OptimizerV2


class Approximation(object):
    """
    """

    def __init__(self, model: Model):
        self.model = model

    def build_logfn(self):
        state, _ = initialize_sampling_state(self.model)
        if not state.all_unobserved_values:
            raise ValueError(
                f"Can not calculate a log probability: the model {self.model.name or ''} has no unobserved values."
            )
        unobserved_keys = state.all_unobserved_values.keys()

        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs)
            _, st = flow.evaluate_model_transformed(self.model, state=st)
            return st.collect_log_prob()

        def vectorize_logp_function(logpfn):
            def vectorized_logpfn(*q_samples):
                return tf.vectorized_map(lambda samples: logpfn(*samples), q_samples)

            return vectorized_logpfn

        return vectorize_logp_function(logpfn)


class MeanField(Approximation):
    @property
    def loc(self):
        pass

    @property
    def cov_matrix(self):
        pass

    def build_posterior(self):
        pass


class FullRank(Approximation):
    @property
    def loc(self):
        pass

    @property
    def cov_matrix(self):
        pass

    def build_posterior(self):
        pass


class LowRank(Approximation):
    @property
    def loc(self):
        pass

    @property
    def cov_matrix(self):
        pass

    def build_posterior(self):
        pass


def fit(
    model: Model,
    *,
    method: str = "advi",
    num_steps: int = 10000,
    sample_size: int = 1,
    random_seed: Optional[int] = None,
    optimizer: Optional[V1_optimizer, V2_optimizer] = None,
    **kwargs,
):
    """
    pass
    """
    if not isinstance(model, Model):
        raise TypeError(
            "`fit` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                type(model)
            )
        )

    _select = dict(advi=MeanField,)

    if isinstance(method, str):
        try:
            inference = _select[method.lower()](model)
        except KeyError:
            raise KeyError("method should be one of %s or Inference instance" % set(_select.keys()))
    elif isinstance(method, Approximation):
        inference = method
    else:
        raise TypeError("method should be one of %s or Inference instance" % set(_select.keys()))

    target_log_prob = inference.build_logfn()
    surrogate_posterior = inference.build_posterior()
    if optimizer:
        opt = optimizer
    else:
        opt = tf.optimizers.Adam(learning_rate=0.1)

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=target_log_prob,
        surrogate_posterior=surrogate_posterior,
        num_steps=num_steps,
        sample_size=sample_size,
        seed=random_seed,
        optimizer=opt,
        **kwargs,
    )

    return losses
