import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        disc = yield pm.Normal("disc", 0, 1)
        return disc

    return simple_model


@pytest.fixture(scope="function")
def model_with_discrete():
    @pm.model()
    def model_with_discrete():
        disc = yield pm.Categorical("disc", probs=[0.1, 0.9])
        return disc

    return model_with_discrete


@pytest.fixture(scope="function")
def model_with_discrete_and_continuous():
    @pm.model()
    def model_with_discrete_and_continuous():
        disc = yield pm.Categorical("disc", probs=[0.1, 0.9])
        norm = yield pm.Normal("mu", 0, 1)
        return norm

    return model_with_discrete_and_continuous


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


@pytest.fixture(scope="module", params=["auto_batch", "trust_manual_batching"], ids=str)
def use_auto_batching_fixture(request):
    return request.param == "auto_batch"


def test_discrete_sampling(model_with_discrete, xla_fixture):
    model = model_with_discrete()
    trace = pm.sample(model=model, sampler_type="compound", xla_fixture=xla_fixture)
    round_value = round(trace.posterior["model_with_discrete/disc"].mean().item(), 1)
    assert round_value == 0.9


def test_compound_sampling(model_with_discrete_and_continuous, xla_fixture):
    model = model_with_discrete_and_continuous()
    trace = pm.sample(model=model, sampler_type="compound", xla_fixture=xla_fixture)
    round_value = round(trace.posterior["model_with_discrete_and_continuous/disc"].mean().item(), 1)
    assert round_value == 0.9
