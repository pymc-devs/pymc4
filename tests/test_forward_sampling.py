import pytest
import re
import numpy as np
import tensorflow as tf
import pymc4 as pm
from pymc4 import forward_sampling


@pytest.fixture(scope="module", params=[(), (1,), (2,), (1, 1), (7, 3)], ids=str)
def sample_shape_fixture(request):
    return request.param


@pytest.fixture(scope="module", params=["NoResampleObserved", "ResampleObserveds"], ids=str)
def sample_from_observed_fixture(request):
    return request.param == "ResampleObserveds"


@pytest.fixture(scope="function")
def model_fixture():
    observed = np.random.randn(10) + 1

    @pm.model
    def model():
        sd = yield pm.HalfNormal("sd", 1.0)
        mu = yield pm.Deterministic("mu", tf.convert_to_tensor(1.0))
        x = yield pm.Normal("x", mu, sd, observed=observed)
        y = yield pm.Normal("y", x, 1e-9)
        dy = yield pm.Deterministic("dy", 2 * y)

    return model, observed


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
