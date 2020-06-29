"""Implements ADVI approximations."""
from typing import Optional, Union
from collections import namedtuple

import arviz as az
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pymc4 import flow
from pymc4.coroutine_model import Model
from pymc4.distributions.transforms import JacobianPreference
from pymc4.inference.utils import initialize_sampling_state
from pymc4.utils import NameParts
from pymc4.variational import updates

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
        raise NotImplementedError

    def _build_posterior(self):
        raise NotImplementedError

    def sample(self, n: int) -> az.InferenceData:
        """Generate samples from posterior distribution."""
        q_samples = dict(zip(self.unobserved_keys, self.approx.sample(n)))

        # TODO - Account for deterministics as well.
        # For all transformed_variables, apply inverse of bijector to sampled values to match support in constraint space.
        _, st = flow.evaluate_model(self.model)
        for transformed_name in self.state.transformed_values:
            untransformed_name = NameParts.from_name(transformed_name).full_untransformed_name
            transform = st.distributions[untransformed_name].transform
            if transform.JacobianPreference == JacobianPreference.Forward:
                q_samples[untransformed_name] = transform.forward(q_samples[transformed_name])
            else:
                q_samples[untransformed_name] = transform.inverse(q_samples[transformed_name])

        # Add a new axis so as n_chains=1 for InferenceData: handles shape issues
        trace = {k: v.numpy()[np.newaxis] for k, v in q_samples.items()}
        trace = az.from_dict(trace, observed_data=self.state.observed_values)
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

    def _build_loc(self, shape, dtype, name):
        loc = tf.Variable(tf.random.normal(shape, seed=self._seed), name=f"{name}/mu", dtype=dtype)
        return loc

    def _build_cov_matrix(self, shape, dtype, name):
        # As per `tfp.vi.fit_surrogate_posterior` docs, use `TransformedVariable` or `DeferredTensor`
        # to ensure all ops invoke gradients while applying transformation.
        scale = tfp.util.TransformedVariable(
            tf.fill(shape, value=tf.constant(1, dtype=dtype)),
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

    def _build_logfn(self):
        """Build flat logp function."""

        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(self.unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs)
            flat_state = dict()
            for key, value in kwargs.items():
                shape = self.state.all_unobserved_values[key].shape
                flat_state[key] = tf.reshape(value, shape)
            _, st = flow.evaluate_meta_model(self.model, values=flat_state)
            return st.collect_log_prob()

        def vectorize_logp_function(logpfn):
            def vectorized_logpfn(*q_samples):
                return tf.vectorized_map(lambda samples: logpfn(*samples), q_samples)

            return vectorized_logpfn

        return vectorize_logp_function(logpfn)

    def _build_loc(self, ndim, dtype, name):
        loc = tf.Variable(tf.random.normal([ndim], seed=self._seed), name=f"{name}/mu", dtype=dtype)
        return loc

    def _build_cov_matrix(self, ndim, dtype, name):
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
            tf.eye(ndim, dtype=dtype), scale_tril, name=f"{name}/sigma"
        )
        return cov_matrix

    def _build_posterior(self):
        def apply_normal(dist_name):
            unobserved_value = self.state.all_unobserved_values[dist_name]
            ndim = np.int32(np.prod(unobserved_value.shape.as_list()))
            dtype = unobserved_value.dtype
            return tfd.MultivariateNormalTriL(
                loc=self._build_loc(ndim, dtype, dist_name),
                scale_tril=self._build_cov_matrix(ndim, dtype, dist_name),
            )

        variational_params = tf.nest.map_structure(apply_normal, self.unobserved_keys)
        return tfd.JointDistributionSequential(variational_params)


class LowRank(Approximation):
    """Low Rank Automatic Differential Variational Inference(Low Rank ADVI)."""

    def _build_logfn(self):
        raise NotImplementedError

    def _build_loc(self):
        raise NotImplementedError

    def _build_cov_matrix(self):
        raise NotImplementedError

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
