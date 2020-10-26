"""Implements ADVI approximations."""
from typing import Optional, Union
from collections import namedtuple

import arviz as az
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util

from pymc4 import flow
from pymc4.coroutine_model import Model
from pymc4.mcmc.utils import initialize_sampling_state
from pymc4.mcmc.samplers import calculate_log_likelihood
from pymc4.utils import NameParts
from pymc4.variational import updates
from pymc4.variational.util import ArrayOrdering

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

        self.order = ArrayOrdering(self.state.all_unobserved_values)
        self.unobserved_keys = self.state.all_unobserved_values.keys()
        (
            self.target_log_prob,
            self.deterministics_callback,
        ) = self._build_logp_and_deterministic_fn()
        self.approx = self._build_posterior()

    def _build_logp_and_deterministic_fn(self):
        """Build vectorized logp and deterministic functions."""

        @tf.function(autograph=False)
        def logpfn(*values):
            split_view = self.order.split(values[0])
            _, st = flow.evaluate_meta_model(self.model, values=split_view)
            return st.collect_log_prob()

        @tf.function(autograph=False)
        def deterministics_callback(q_samples):
            st = flow.SamplingState.from_values(
                q_samples, observed_values=self.state.observed_values
            )
            _, st = flow.evaluate_model_transformed(self.model, state=st)
            for transformed_name in st.transformed_values:
                untransformed_name = NameParts.from_name(transformed_name).full_untransformed_name
                st.deterministics_values[untransformed_name] = st.untransformed_values.pop(
                    untransformed_name
                )
            return st.deterministics_values

        def vectorize_function(function):
            def vectorizedfn(*q_samples):
                return tf.vectorized_map(lambda samples: function(*samples), q_samples)

            return vectorizedfn

        return vectorize_function(logpfn), vectorize_function(deterministics_callback)

    def _build_posterior(self):
        raise NotImplementedError

    def sample(self, n: int = 500, include_log_likelihood: bool = False) -> az.InferenceData:
        """Generate samples from posterior distribution."""
        samples = self.approx.sample(n)
        q_samples = self.order.split_samples(samples, n)
        q_samples = dict(**q_samples, **self.deterministics_callback(q_samples))

        # Add a new axis so as n_chains=1 for InferenceData: handles shape issues
        trace = {k: v.numpy()[np.newaxis] for k, v in q_samples.items()}
        log_likelihood_dict = dict()
        if include_log_likelihood:
            log_likelihood_dict = calculate_log_likelihood(self.model, trace, self.state)

        trace = az.from_dict(
            trace,
            observed_data=self.state.observed_values,
            log_likelihood=log_likelihood_dict if include_log_likelihood else None,
        )
        return trace


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

    def _build_posterior(self):
        flattened_shape = self.order.size
        dtype = dtype_util.common_dtype(
            self.state.all_unobserved_values.values(), dtype_hint=tf.float64
        )
        loc = tf.Variable(tf.random.normal([flattened_shape], dtype=dtype), name="mu")
        cov_param = tfp.util.TransformedVariable(
            tf.ones(flattened_shape, dtype=dtype), tfb.Softplus(), name="sigma"
        )
        advi_approx = tfd.MultivariateNormalDiag(loc=loc, scale_diag=cov_param)
        return advi_approx


class FullRank(Approximation):
    """
    Full Rank ADVI.

    This class implements Full Rank Automatic Differentiation Variational Inference. It posits Multivariate
    Gaussian family to fit posterior. And estimates a full covariance matrix. As a result, it comes with
    higher computation costs.

    References
    ----------
    -   Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
    and Blei, D. M. (2016). Automatic Differentiation Variational
    Inference. arXiv preprint arXiv:1603.00788.
    """

    def _build_posterior(self):
        flattened_shape = self.order.size
        dtype = dtype_util.common_dtype(
            self.state.all_unobserved_values.values(), dtype_hint=tf.float64
        )
        loc = tf.Variable(tf.random.normal([flattened_shape], dtype=dtype), name="mu")
        scale_tril = tfb.FillScaleTriL(
            diag_bijector=tfb.Chain(
                [
                    tfb.Shift(tf.cast(1e-3, dtype)),  # diagonal offset
                    tfb.Softplus(),
                    tfb.Shift(tf.cast(np.log(np.expm1(1.0)), dtype)),  # initial scale
                ]
            ),
            diag_shift=None,
        )

        cov_matrix = tfp.util.TransformedVariable(
            tf.eye(flattened_shape, dtype=dtype), scale_tril, name="sigma"
        )
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=cov_matrix)


class LowRank(Approximation):
    """Low Rank Automatic Differential Variational Inference(Low Rank ADVI)."""

    def _build_posterior(self):
        raise NotImplementedError


def fit(
    model: Optional[Model] = None,
    method: Union[str, MeanField, FullRank] = "advi",
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
    optimizer : TF1-style | TF2-style | from pymc4/variational/updates
        Tensorflow optimizer to use
    kwargs : Optional[Dict[str, Any]]
        Pass extra non-default arguments to
        ``tensorflow_probability.vi.fit_surrogate_posterior``

    Returns
    -------
    ADVIFit : collections.namedtuple
        Named tuple, including approximation, ELBO losses depending on the `trace_fn`
    """
    _select = dict(advi=MeanField, fullrank_advi=FullRank)

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

    # Defining `opt = optimizer or updates.adam()`
    # leads to optimizer initialization issues from tf.
    if optimizer:
        opt = optimizer
    else:
        opt = updates.adam()

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
