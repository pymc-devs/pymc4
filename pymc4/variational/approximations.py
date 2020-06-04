from typing import Optional, Union

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

    def __init__(self, model: Model, random_seed: Optional[int] = None):
        self.model = model
        self._seed = random_seed
        self.state, self.deterministic_names = initialize_sampling_state(model)
        if not self.state.all_unobserved_values:
            raise ValueError(
                f"Can not calculate a log probability: the model {model.name or ''} has no unobserved values."
            )

        self.unobserved_keys, self.unobserved_values = zip(*self.state.all_unobserved_values.items())
        self.target_log_prob = self._build_logfn()
        self.approx = self._build_posterior()

    def _build_logfn(self):
        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(self.unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs)
            _, st = flow.evaluate_model_transformed(self.model, state=st)
            return st.collect_log_prob()

        def vectorize_logp_function(logpfn):
            def vectorized_logpfn(*q_samples):
                return tf.vectorized_map(lambda samples: logpfn(*samples), q_samples)

            return vectorized_logpfn

        return vectorize_logp_function(logpfn)

    def flatten_view(self):
        pass

    def _build_posterior(self):
        raise NotImplementedError


class MeanField(Approximation):
    def _build_loc(self, shape, dtype):
        loc = tf.Variable(tf.random.normal(shape), dtype=dtype)
        return loc

    def _build_cov_matrix(self, shape, dtype):
        scale = tfp.util.TransformedVariable(
            tf.fill(shape, value=tf.constant(0.02, dtype=dtype)),
            tfb.Softplus(),  # For positive values of scale
        )
        return scale

    def _build_posterior(self):
        def apply_normal(param):
            shape = param.shape
            dtype = param.dtype
            return tfd.Normal(self._build_loc(shape, dtype), self._build_cov_matrix(shape, dtype))

        variational_params = tf.nest.map_structure(apply_normal, self.unobserved_values)
        return tfd.JointDistributionSequential(variational_params)


class FullRank(Approximation):
    def _build_loc(self):
        pass

    def _build_cov_matrix(self):
        pass

    def _build_posterior(self):
        pass


class LowRank(Approximation):
    def _build_loc(self):
        pass

    def _build_cov_matrix(self):
        pass

    def _build_posterior(self):
        pass


def fit(
    model: Model,
    *,
    method: str = "advi",
    num_steps: int = 10000,
    sample_size: int = 1,
    random_seed: Optional[int] = None,
    optimizer: Union[V1_optimizer, V2_optimizer, None] = None,
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
            inference = _select[method.lower()](model, random_seed)
        except KeyError:
            raise KeyError(
                "method should be one of %s or Approximation instance" % set(_select.keys())
            )
    elif isinstance(method, Approximation):
        inference = method
    else:
        raise TypeError(
            "method should be one of %s or Approximation instance" % set(_select.keys())
        )

    if optimizer:
        opt = optimizer
    else:
        opt = tf.optimizers.Adam(learning_rate=0.1)

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=inference.target_log_prob,
        surrogate_posterior=inference.approx,
        num_steps=num_steps,
        sample_size=sample_size,
        seed=random_seed,
        optimizer=opt,
        **kwargs,
    )

    return losses
