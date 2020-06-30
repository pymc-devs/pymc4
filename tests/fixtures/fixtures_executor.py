"""
Fixtures for test_executor.py
"""
import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf

from pymc4 import distributions as dist


TEST_SHAPES = [(), (1,), (3,), (1, 1), (1, 3), (5, 3)]


@pytest.fixture(scope="module", params=TEST_SHAPES, ids=str)
def batch_shapes(request):
    return request.param


@pytest.fixture(scope="module", params=TEST_SHAPES, ids=str)
def sample_shapes(request):
    return request.param


@pytest.fixture(scope="module")
def distribution_parameters(batch_shapes, sample_shapes):
    observed = np.random.randn(*(sample_shapes + batch_shapes))
    return batch_shapes, observed


@pytest.fixture(scope="module", params=["decorate_model", "use_plain_function"], ids=str)
def pm_model_decorate(request):
    return request.param == "decorate_model"


@pytest.fixture(scope="module")
def complex_model():
    @pm.model
    def nested_model(cond):
        norm = yield dist.HalfNormal("n", cond ** 2, transform=dist.transforms.Log())
        return norm

    @pm.model(keep_return=False)
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    return complex_model


@pytest.fixture(scope="module")
def complex_model_with_observed():
    @pm.model
    def nested_model(cond):
        norm = yield dist.HalfNormal(
            "n", cond ** 2, observed=np.ones(10), transform=dist.transforms.Log()
        )
        return norm

    @pm.model(keep_return=False)
    def complex_model():
        norm = yield dist.Normal("n", 0, 1)
        result = yield nested_model(norm, name="a")
        return result

    return complex_model


@pytest.fixture(scope="module")
def transformed_model():
    def transformed_model():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log())
        return norm

    return transformed_model


@pytest.fixture(scope="module")
def transformed_model_with_observed():
    def transformed_model_with_observed():
        norm = yield dist.HalfNormal("n", 1, transform=dist.transforms.Log(), observed=1.0)
        return norm

    return transformed_model_with_observed


@pytest.fixture(scope="module")
def fixture_model_with_stacks(distribution_parameters, pm_model_decorate):
    batch_shape, observed = distribution_parameters
    expected_obs_shape = (
        ()
        if isinstance(observed, float)
        else observed.shape[: len(observed.shape) - len(batch_shape)]
    )
    if pm_model_decorate:
        expected_rv_shapes = {"model/loc": (), "model/obs": expected_obs_shape}
    else:
        expected_rv_shapes = {"loc": (), "obs": expected_obs_shape}

    def model():
        loc = yield pm.Normal("loc", 0, 1)
        obs = yield pm.Normal("obs", loc, 1, event_stack=batch_shape, observed=observed)
        return obs

    if pm_model_decorate:
        model = pm.model(model)

    return model, expected_rv_shapes


@pytest.fixture(scope="module")
def model_with_deterministics():
    expected_deterministics = ["model/abs_norm", "model/sine_norm", "model/norm_copy"]
    expected_ops = [np.abs, np.sin, lambda x: x]
    expected_ops_inputs = [["model/norm"], ["model/norm"], ["model/norm"]]

    @pm.model
    def model():
        norm = yield dist.Normal("norm", 0, 1)
        abs_norm = yield dist.Deterministic("abs_norm", tf.abs(norm))
        sine_norm = yield dist.Deterministic("sine_norm", tf.sin(norm))
        norm_copy = yield dist.Deterministic("norm_copy", norm)
        obs = yield dist.Normal("obs", 0, abs_norm)

    return model, expected_deterministics, expected_ops, expected_ops_inputs


# TODO - lots of code duplication with fixtures_models.deterministics_in_nested_models
@pytest.fixture(scope="function")
def deterministics_in_nested_models():
    @pm.model
    def nested_model(cond):
        x = yield pm.Normal("x", cond, 1)
        dx = yield pm.Deterministic("dx", x + 1)
        return dx

    @pm.model
    def outer_model():
        cond = yield pm.HalfNormal("cond", 1, conditionally_independent=True)
        dcond = yield pm.Deterministic("dcond", cond * 2)
        dx = yield nested_model(dcond)
        ddx = yield pm.Deterministic("ddx", dx)
        return ddx

    expected_untransformed = {"outer_model/cond", "outer_model/nested_model/x"}
    expected_transformed = {"outer_model/__log_cond"}
    expected_deterministics = {
        "outer_model/dcond",
        "outer_model/ddx",
        "outer_model/nested_model/dx",
        "outer_model/nested_model",
        "outer_model",
    }
    deterministic_mapping = {
        "outer_model/dcond": (["outer_model/cond"], lambda x: x * 2),
        "outer_model/ddx": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model/nested_model": (["outer_model/nested_model/x"], lambda x: x + 1),
        "outer_model": (["outer_model/nested_model/x"], lambda x: x + 1),
    }

    return (
        outer_model,
        expected_untransformed,
        expected_transformed,
        expected_deterministics,
        deterministic_mapping,
    )
