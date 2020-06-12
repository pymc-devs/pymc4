"""Implements ADVI approximations."""
from typing import Optional, Union
from collections import namedtuple

# import arviz as az
import tensorflow as tf
import tensorflow_probability as tfp

from pymc4 import flow
from pymc4.coroutine_model import Model
from pymc4.inference.utils import initialize_sampling_state

tfd = tfp.distributions
tfb = tfp.bijectors
ADVIFit = namedtuple("ADVIFit", "approximation, losses")


class Approximation(tf.Module):
    """Base Approximation class."""

    def __init__(self, model: Optional[Model] = None, random_seed: Optional[int] = None):
        if not isinstance(model, Model):
            raise TypeError(
                "`fit` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                    type(model)
                )
            )

        self.model = model
        self._seed = random_seed
        self.state, self.deterministic_names = initialize_sampling_state(model)
        if not self.state.all_unobserved_values:
            raise ValueError(
                f"Can not calculate a log probability: the model {model.name or ''} has no unobserved values."
            )

        self.unobserved_keys = self.state.all_unobserved_values.keys()
        self.target_log_prob = self._build_logfn()
        self.approx = self._build_posterior()

    def _build_logfn(self):
        """Build vectorized logp function."""

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

    def _build_posterior(self):
        raise NotImplementedError

    def flatten_view(self):
        """Flattened view of the variational parameters."""
        pass

    def sample(self, n):
        """Generate samples from posterior distribution."""
        q_samples = dict(zip(self.unobserved_keys, self.approx.sample(n, seed=self._seed)))
        # trace = az.from_dict(q_samples, observed_data=self.state.observed_values)
        return q_samples


class MeanField(Approximation):
    """
    Mean Field ADVI.

    This class implements Mean Field Automatic Differentiation Variational Inference. It posits spherical 
    Gaussian family to fit posterior. And assumes the parameters to be uncorrelated.

    References
    ----------
    -   Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
    and Blei, D. M. (2016). Automatic Differentiation Variational
    Inference. arXiv preprint arXiv:1603.00788.
    """

    def _build_loc(self, shape, dtype, name):
        loc = tf.Variable(tf.random.normal(shape, seed=self._seed), name=f"{name}/mu", dtype=dtype)
        return loc

    def _build_cov_matrix(self, shape, dtype, name):
        # As per `tfp.vi.fit_surrogate_posterior` docs, use `TransformedVariable` or `DeferredTensor`
        # to ensure all ops invoke gradients while applying transformation.
        scale = tfp.util.TransformedVariable(
            tf.fill(shape, value=tf.constant(0.02, dtype=dtype)),
            tfb.Softplus(),  # For positive values of scale
            name=f"{name}/sigma",
        )
        return scale

    def _build_posterior(self):
        def apply_normal(dist_name):
            unobserved_value = self.state.all_unobserved_values[dist_name]
            shape = unobserved_value.shape
            dtype = unobserved_value.dtype
            return tfd.Normal(
                self._build_loc(shape, dtype, dist_name),
                self._build_cov_matrix(shape, dtype, dist_name),
            )

        # Should we use `tf.nest.map_structure` or `pm.utils.map_structure`?
        variational_params = tf.nest.map_structure(apply_normal, self.unobserved_keys)
        return tfd.JointDistributionSequential(variational_params)


class FullRank(Approximation):
    """Full Rank Automatic Differential Variational Inference(Full Rank ADVI)."""

    def _build_loc(self):
        pass

    def _build_cov_matrix(self):
        pass

    def _build_posterior(self):
        pass


class LowRank(Approximation):
    """Low Rank Automatic Differential Variational Inference(Low Rank ADVI)."""

    def _build_loc(self):
        pass

    def _build_cov_matrix(self):
        pass

    def _build_posterior(self):
        pass


def fit(
    model: Optional[Model] = None,
    method: Union[str, MeanField] = "advi",
    num_steps: int = 10000,
    sample_size: int = 1,
    random_seed: Optional[int] = None,
    optimizer=None,
    **kwargs,
):
    """
    Fit an approximating distribution to log_prob of the model.

    Parameters
    ----------
    model : Optional[:class:`Model`]
        Model to fit posterior against
    method : Union[str, :class:`Approximation`]
        Method to fit model using VI

        - 'advi' for :class:`MeanField`
        - 'fullrank_advi' for :class:`FullRank`
        - 'lowrank_advi' for :class:`LowRank`
        - or directly pass in :class:`Approximation` instance
    num_steps : int
        Number of iterations to run the optimizer
    sample_size : int
        Number of Monte Carlo samples used for approximation
    random_seed : Optional[int]
        Seed for tensorflow random number generator
    optimizer : TF1-style or TF2-style optimizer
        Tensorflow optimizer to use
    kwargs : Optional[Dict[str, Any]]
        Pass extra non-default arguments to
        ``tensorflow_probability.vi.fit_surrogate_posterior``

    Returns
    -------
    ADVIFit : collections.namedtuple
        Named tuple, including approximation, ELBO losses depending on the `trace_fn`
    """
    _select = dict(advi=MeanField,)

    if isinstance(method, str):
        # Here we assume that `model` parameter is provided by the user.
        try:
            inference = _select[method.lower()](model, random_seed)
        except KeyError:
            raise KeyError(
                "method should be one of %s or Approximation instance" % set(_select.keys())
            )

    elif isinstance(method, Approximation):
        # Here we assume that `model` parameter is not provided by the user
        # as the :class:`Approximation` itself contains :class:`Model` instance.
        inference = method

    else:
        raise TypeError(
            "method should be one of %s or Approximation instance" % set(_select.keys())
        )

    if optimizer:
        opt = optimizer
    else:
        opt = tf.optimizers.Adam(learning_rate=0.1)

    @tf.function(autograph=False)
    def run_approximation():
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

    return ADVIFit(inference, run_approximation())
