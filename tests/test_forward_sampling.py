import pytest
import re
import collections
import numpy as np
import tensorflow as tf
import pymc4 as pm
from pymc4 import forward_sampling
from pymc4.flow.executor import EvaluationError

from .fixtures.fixtures_sampling import (
    sample_shape,
    vectorized_model_fixture,
    forward_sampling_core_shapes,
    sample_from_observed,
    model_fixture,
    model_with_observed,
    posterior_predictive,
    glm_model,
)


def test_sample_prior_predictive(model_fixture, sample_shape, sample_from_observed):
    model, observed = model_fixture

    prior = pm.sample_prior_predictive(
        model(), sample_shape[1:], sample_from_observed
    ).prior_predictive
    if sample_from_observed:
        assert set(["model/sd", "model/x", "model/y", "model/mu", "model/dy"]) == set(
            prior.data_vars
        )
        assert all((value.shape == sample_shape for value in prior.values()))
    else:
        assert set(["model/sd", "model/y", "model/mu", "model/dy"]) == set(prior.data_vars)
        assert all(
            (
                value.shape == sample_shape
                for name, value in prior.items()
                if name in {"model/sd", "model/mu"}
            )
        )
        assert all(
            (
                value.shape == sample_shape + observed.shape
                for name, value in prior.items()
                if name in {"model/x", "model/y", "model/dy"}
            )
        )
        assert np.allclose(prior["model/y"], observed, rtol=1e-5)
        assert np.allclose(prior["model/y"] * 2, prior["model/dy"])


def test_sample_prior_predictive_var_names(model_fixture):
    model, observed = model_fixture

    prior = pm.sample_prior_predictive(
        model(), var_names=["model/sd"], sample_shape=()
    ).prior_predictive
    assert set(prior) == set(["model/sd"])

    prior = pm.sample_prior_predictive(
        model(), var_names=["model/x", "model/y"], sample_shape=()
    ).prior_predictive
    assert set(prior) == set(["model/x", "model/y"])

    # Assert we can get the values of observeds if we ask for them explicitly
    # even if sample_from_observed is False
    prior = pm.sample_prior_predictive(
        model(), var_names=["model/x", "model/y"], sample_shape=(), sample_from_observed=False
    ).prior_predictive
    assert set(prior) == set(["model/x", "model/y"])
    assert np.all(prior["model/x"] == observed)

    # Assert an exception is raised if we pass wrong names
    model_func = model()
    expected_message = "Some of the supplied var_names are not defined in the supplied model {}.\nList of unknown var_names: {}".format(
        model_func, ["X"]
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        prior = pm.sample_prior_predictive(model_func, var_names=["X", "model/y"], sample_shape=())


def test_sample_prior_predictive_int_sample_shape(model_fixture, n_draws):
    model, observed = model_fixture

    prior_int = forward_sampling.sample_prior_predictive(model(), sample_shape=n_draws)

    prior_tuple = forward_sampling.sample_prior_predictive(model(), sample_shape=(n_draws,))

    assert all(
        (
            prior_int.prior_predictive[k].shape == v.shape
            for k, v in prior_tuple.prior_predictive.items()
        )
    )


def test_posterior_predictive_executor(model_with_observed):
    model, observed, core_ppc_shapes, _ = model_with_observed
    _, prior_state = pm.evaluate_model_transformed(model(), observed=observed)
    _, ppc_state = pm.evaluate_model_posterior_predictive(model(), observed=observed)

    # Assert that a normal evaluation has all observeds and the values match
    # to the observations
    assert len(prior_state.observed_values) == 3
    for var, val in observed.items():
        assert np.all(prior_state.all_values[var] == val)

    # Assert that a posterior predictive evaluation has no observed values
    # but the shapes of the samples match the supplied observed shapes
    assert len(ppc_state.observed_values) == 0
    for var, shape in core_ppc_shapes.items():
        assert (
            collections.ChainMap(ppc_state.all_values, ppc_state.deterministics)[var].numpy().shape
            == shape
        )
        if var in observed:
            assert np.any(ppc_state.all_values[var] != val for val in observed.values())


def test_sample_posterior_predictive(posterior_predictive):
    (
        model,
        observed,
        core_ppc_shapes,
        observed_in_RV,
        trace,
        num_samples,
        num_chains,
    ) = posterior_predictive

    if observed_in_RV:
        observed_kwarg = None
    else:
        observed_kwarg = observed
    ppc = pm.sample_posterior_predictive(
        model(), trace, observed=observed_kwarg
    ).posterior_predictive
    assert set(sorted(list(ppc))) == set(observed)
    assert np.all(
        [v.shape == (num_chains, num_samples) + observed[k].shape for k, v in ppc.items()]
    )


def test_sample_ppc_var_names(model_fixture):
    model, observed = model_fixture
    trace = pm.inference.utils.trace_to_arviz(
        {
            "model/sd": tf.ones((10, 1), dtype="float32"),
            "model/y": tf.convert_to_tensor(observed[:, None]),
        }
    )

    with pytest.raises(ValueError):
        forward_sampling.sample_posterior_predictive(model(), trace, var_names=[])

    with pytest.raises(KeyError):
        forward_sampling.sample_posterior_predictive(
            model(), trace, var_names=["name not in model!"]
        )

    with pytest.raises(TypeError):
        trace.posterior["name not in model!"] = tf.constant(1.0)
        pm.sample_posterior_predictive(model(), trace)
    del trace.posterior["name not in model!"]

    var_names = ["model/sd", "model/x", "model/dy"]
    ppc = pm.sample_posterior_predictive(model(), trace, var_names=var_names).posterior_predictive
    assert set(var_names) == set(ppc)
    assert ppc["model/sd"].shape == trace.posterior["model/sd"].shape


def test_sample_ppc_corrupt_trace():
    @pm.model
    def model():
        x = yield pm.Normal("x", tf.ones(5), 1, reinterpreted_batch_ndims=1)
        y = yield pm.Normal("y", x, 1)

    trace1 = pm.inference.utils.trace_to_arviz({"model/x": tf.ones((7, 1), dtype="float32")})

    trace2 = pm.inference.utils.trace_to_arviz(
        {"model/x": tf.ones((1, 5), dtype="float32"), "model/y": tf.zeros((1, 1), dtype="float32")}
    )
    with pytest.raises(EvaluationError):
        forward_sampling.sample_posterior_predictive(model(), trace1)
    with pytest.raises(EvaluationError):
        forward_sampling.sample_posterior_predictive(model(), trace2)


def test_vectorized_sample_prior_predictive(
    vectorized_model_fixture, use_auto_batching, forward_sampling_core_shapes, sample_shape
):
    model, is_vectorized_model = vectorized_model_fixture
    core_shapes = forward_sampling_core_shapes
    prior = forward_sampling.sample_prior_predictive(
        model(), sample_shape=sample_shape, use_auto_batching=use_auto_batching
    ).prior_predictive
    if not use_auto_batching and not is_vectorized_model and len(sample_shape) > 0:
        with pytest.raises(AssertionError):
            for k, v in core_shapes.items():
                # The (1,) comes from trace_to_arviz imposed chain axis
                assert prior[k].shape == (1,) + sample_shape + v
    else:
        for k, v in core_shapes.items():
            # The (1,) comes from trace_to_arviz imposed chain axis
            assert prior[k].shape == (1,) + sample_shape + v


def test_sample_prior_predictive_on_glm(glm_model, use_auto_batching, sample_shape):
    model, is_vectorized_model, core_shapes = glm_model
    if not use_auto_batching and not is_vectorized_model and len(sample_shape) > 0:
        with pytest.raises(AssertionError):
            prior = forward_sampling.sample_prior_predictive(
                model(), sample_shape=sample_shape, use_auto_batching=use_auto_batching,
            ).prior_predictive
            for k, v in core_shapes.items():
                # The (1,) comes from trace_to_arviz imposed chain axis
                assert prior[k].shape == (1,) + sample_shape + v
    else:
        prior = forward_sampling.sample_prior_predictive(
            model(), sample_shape=sample_shape, use_auto_batching=use_auto_batching
        ).prior_predictive
        for k, v in core_shapes.items():
            # The (1,) comes from trace_to_arviz imposed chain axis
            assert prior[k].shape == (1,) + sample_shape + v


def test_vectorized_sample_posterior_predictive(
    vectorized_model_fixture, use_auto_batching, forward_sampling_core_shapes, sample_shape
):
    model, is_vectorized_model = vectorized_model_fixture
    core_shapes = forward_sampling_core_shapes
    trace = pm.inference.utils.trace_to_arviz(
        {
            # The transposition of the first two axis comes from trace_to_arviz
            # that does this to the output of `sample` to get (num_chains, num_samples, ...)
            # instead of (num_samples, num_chains, ...)
            k: tf.zeros((sample_shape[1], sample_shape[0]) + sample_shape[2:] + v)
            for k, v in core_shapes.items()
            if k not in ["model/x"]
        }
    )
    if not use_auto_batching and not is_vectorized_model and len(sample_shape) > 0:
        with pytest.raises((ValueError, EvaluationError)):
            # This can raise ValueError when tfp distributions complain about
            # the parameter shapes being imcompatible or it can raise an
            # EvaluationError because the distribution shape is not compatible
            # with the supplied observations
            forward_sampling.sample_posterior_predictive(
                model(), trace=trace, use_auto_batching=use_auto_batching
            )
    else:
        ppc = forward_sampling.sample_posterior_predictive(
            model(), trace=trace, use_auto_batching=use_auto_batching
        ).posterior_predictive
        for k, v in ppc.items():
            assert v.shape == sample_shape + core_shapes[k]


def test_sample_posterior_predictive_on_glm(glm_model, use_auto_batching, sample_shape):
    model, is_vectorized_model, core_shapes = glm_model
    trace = pm.inference.utils.trace_to_arviz(
        {
            # The transposition of the first two axis comes from trace_to_arviz
            # that does this to the output of `sample` to get (num_chains, num_samples, ...)
            # instead of (num_samples, num_chains, ...)
            k: tf.zeros((sample_shape[1], sample_shape[0]) + sample_shape[2:] + v)
            for k, v in core_shapes.items()
            if k not in ["model/y"]
        }
    )
    if (
        not use_auto_batching
        and not is_vectorized_model
        and (sample_shape not in [(), (1,), (1, 1)]) > 0
    ):
        with pytest.raises(Exception):
            # This can raise many types of Exceptions.
            # For example, ValueError when tfp distributions complain about
            # the parameter shapes being imcompatible or it can raise an
            # EvaluationError because the distribution shape is not compatible
            # with the supplied observations. Also, if in a @tf.function,
            # it can raise InvalidArgumentError.
            # Furthermore, in some cases, sampling may exit without errors, but
            # the resulting shapes will be wrong
            ppc = forward_sampling.sample_posterior_predictive(
                model(), trace=trace, use_auto_batching=use_auto_batching
            )
            for k, v in ppc.items():
                assert v.shape == sample_shape + core_shapes[k]
    else:
        ppc = forward_sampling.sample_posterior_predictive(
            model(), trace=trace, use_auto_batching=use_auto_batching
        ).posterior_predictive
        for k, v in ppc.items():
            assert v.shape == sample_shape + core_shapes[k]


def test_posterior_predictive_on_root_variable(use_auto_batching):
    n_obs = 5
    n_samples = 6
    n_chains = 4

    @pm.model
    def model():
        x = yield pm.Normal(
            "x",
            np.zeros(n_obs, dtype="float32"),
            1,
            observed=np.zeros(n_obs, dtype="float32"),
            conditionally_independent=True,
            reinterpreted_batch_ndims=1,
        )
        beta = yield pm.Normal("beta", 0, 1, conditionally_independent=True)
        bias = yield pm.Normal("bias", 0, 1, conditionally_independent=True)
        mu = beta[..., None] * x + bias[..., None]
        yield pm.Normal(
            "obs", mu, 1, observed=np.ones(n_obs, dtype="float32"), reinterpreted_batch_ndims=1
        )

    trace = pm.inference.utils.trace_to_arviz(
        {
            "model/beta": tf.zeros((n_samples, n_chains), dtype="float32"),
            "model/bias": tf.zeros((n_samples, n_chains), dtype="float32"),
        }
    )
    ppc = forward_sampling.sample_posterior_predictive(
        model(), trace=trace, use_auto_batching=use_auto_batching
    ).posterior_predictive
    if not use_auto_batching:
        _, state = pm.evaluate_model_posterior_predictive(
            model(), sample_shape=(n_chains, n_samples)
        )
        assert state.untransformed_values["model/x"].numpy().shape == (n_chains, n_samples, n_obs)
    assert ppc["model/obs"].shape == (n_chains, n_samples, n_obs)
    assert ppc["model/x"].shape == (n_chains, n_samples, n_obs)
