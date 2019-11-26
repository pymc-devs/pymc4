import pytest
import re
import itertools
import collections
import numpy as np
import tensorflow as tf
import pymc4 as pm
from pymc4 import forward_sampling


@pytest.fixture(scope="module", params=[(), (1,), (2,), (1, 1), (7, 3)], ids=str)
def sample_shape_fixture(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 10, 100], ids=str)
def n_draws_fixture(request):
    return request.param


@pytest.fixture(scope="module", params=["NoResampleObserved", "ResampleObserveds"], ids=str)
def sample_from_observed_fixture(request):
    return request.param == "ResampleObserveds"


@pytest.fixture(scope="function")
def model_fixture():
    observed = np.random.randn(10).astype("float32") + 1

    @pm.model
    def model():
        sd = yield pm.HalfNormal("sd", 1.0)
        mu = yield pm.Deterministic("mu", tf.convert_to_tensor(1.0))
        x = yield pm.Normal("x", mu, sd, observed=observed)
        y = yield pm.Normal("y", x, 1e-9)
        dy = yield pm.Deterministic("dy", 2 * y)

    return model, observed


@pytest.fixture(scope="module", params=["observed_in_RV", "observed_in_eval"])
def model_with_observed_fixture(request):
    observed_in_RV = request.param == "observed_in_RV"
    observed = {
        "model/x": np.ones(10, dtype="float32"),
        "model/y": np.ones(1, dtype="float32"),
        "model/z": np.ones((10, 10), dtype="float32"),
    }
    observed_kwargs = {k: v if observed_in_RV else None for k, v in observed.items()}
    core_ppc_shapes = {
        "model/sd": (),
        "model/x": (10,),
        "model/y": (1,),
        "model/d": (10,),
        "model/z": (10, 10),
        "model/u": (10, 10),
    }

    @pm.model
    def model():
        sd = yield pm.Exponential("sd", 1)
        x = yield pm.Normal("x", 0, 1, observed=observed_kwargs["model/x"])
        y = yield pm.HalfNormal("y", 1, observed=observed_kwargs["model/y"])
        d = yield pm.Deterministic("d", x + y)
        z = yield pm.Normal("z", d, sd, observed=observed_kwargs["model/z"])
        u = yield pm.Exponential("u", z)
        return u

    return model, observed, core_ppc_shapes, observed_in_RV


@pytest.fixture(scope="module")
def posterior_predictive_fixture(model_with_observed_fixture):
    num_samples = 40
    num_chains = 3
    (model, observed, core_ppc_shapes, observed_in_RV,) = model_with_observed_fixture
    trace, _ = pm.inference.sampling.sample(
        model(), num_samples=num_samples, num_chains=num_chains, observed=observed,
    )
    return (
        model,
        observed,
        core_ppc_shapes,
        observed_in_RV,
        trace,
        num_samples,
        num_chains,
    )


def test_sample_prior_predictive(model_fixture, sample_shape_fixture, sample_from_observed_fixture):
    model, observed = model_fixture

    prior = forward_sampling.sample_prior_predictive(
        model(), sample_shape_fixture, sample_from_observed_fixture
    )
    if sample_from_observed_fixture:
        assert set(["model/sd", "model/x", "model/y", "model/mu", "model/dy"]) == set(prior)
        assert all((value.shape == sample_shape_fixture for value in prior.values()))
    else:
        assert set(["model/sd", "model/y", "model/mu", "model/dy"]) == set(prior)
        assert all(
            (
                value.shape == sample_shape_fixture
                for name, value in prior.items()
                if name in {"model/sd", "model/mu"}
            )
        )
        assert all(
            (
                value.shape == sample_shape_fixture + observed.shape
                for name, value in prior.items()
                if name in {"model/x", "model/y", "model/dy"}
            )
        )
        assert np.allclose(prior["model/y"], observed, rtol=1e-5)
        assert np.allclose(prior["model/y"] * 2, prior["model/dy"])


def test_sample_prior_predictive_var_names(model_fixture):
    model, observed = model_fixture

    prior = forward_sampling.sample_prior_predictive(
        model(), var_names=["model/sd"], sample_shape=(),
    )
    assert set(prior) == set(["model/sd"])

    prior = forward_sampling.sample_prior_predictive(
        model(), var_names=["model/x", "model/y"], sample_shape=(),
    )
    assert set(prior) == set(["model/x", "model/y"])

    # Assert we can get the values of observeds if we ask for them explicitly
    # even if sample_from_observed_fixture is False
    prior = forward_sampling.sample_prior_predictive(
        model(), var_names=["model/x", "model/y"], sample_shape=(), sample_from_observed=False
    )
    assert set(prior) == set(["model/x", "model/y"])
    assert np.all(prior["model/x"] == observed)

    # Assert an exception is raised if we pass wrong names
    model_func = model()
    expected_message = "Some of the supplied var_names are not defined in the supplied model {}.\nList of unknown var_names: {}".format(
        model_func, ["X"]
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        prior = forward_sampling.sample_prior_predictive(
            model_func, var_names=["X", "model/y"], sample_shape=(),
        )


def test_sample_prior_predictive_int_sample_shape(model_fixture, n_draws_fixture):
    model, observed = model_fixture

    prior_int = forward_sampling.sample_prior_predictive(model(), sample_shape=n_draws_fixture)

    prior_tuple = forward_sampling.sample_prior_predictive(model(), sample_shape=(n_draws_fixture,))

    assert all((prior_int[k].shape == v.shape for k, v in prior_tuple.items()))


def test_posterior_predictive_executor(model_with_observed_fixture):
    model, observed, core_ppc_shapes, _ = model_with_observed_fixture
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
            assert np.any(ppc_state.all_values[var] != val)


def test_sample_posterior_predictive(posterior_predictive_fixture):
    (
        model,
        observed,
        core_ppc_shapes,
        observed_in_RV,
        trace,
        num_samples,
        num_chains,
    ) = posterior_predictive_fixture

    if observed_in_RV:
        observed_kwarg = None
    else:
        observed_kwarg = observed
    ppc = pm.sample_posterior_predictive(model(), trace, observed=observed_kwarg)
    assert set(sorted(list(ppc))) == set(observed)
    assert np.all(
        [v.shape == (num_samples, num_chains) + observed[k].shape for k, v in ppc.items()]
    )


def test_sample_ppc_var_names(model_fixture):
    model, observed = model_fixture
    trace = {
        "model/sd": tf.convert_to_tensor(np.array(1.0, dtype="float32")),
        "model/y": tf.convert_to_tensor(observed),
    }

    with pytest.raises(ValueError):
        pm.sample_posterior_predictive(model(), trace, var_names=[])

    with pytest.raises(KeyError):
        pm.sample_posterior_predictive(model(), trace, var_names=["name not in model!"])

    with pytest.raises(TypeError):
        bad_trace = trace.copy()
        bad_trace["name not in model!"] = tf.constant(1.0)
        pm.sample_posterior_predictive(model(), bad_trace)

    var_names = ["model/sd", "model/x", "model/dy"]
    ppc = pm.sample_posterior_predictive(model(), trace, var_names=var_names)
    assert set(var_names) == set(ppc)
    assert ppc["model/sd"].shape == trace["model/sd"].shape
    assert np.all([v.shape == observed.shape for k, v in ppc.items() if k != "model/sd"])
