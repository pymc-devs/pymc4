import pytest
import pymc4 as pm
import numpy as np


@pytest.fixture(scope="function")
def model_with_discrete_categorical():
    @pm.model()
    def model_with_discrete_categorical():
        disc = yield pm.Categorical("disc", probs=[0.1, 0.9])
        return disc

    return model_with_discrete_categorical


@pytest.fixture(scope="function")
def model_with_discrete_bernoulli():
    @pm.model()
    def model_with_discrete_bernoulli():
        disc = yield pm.Bernoulli("disc", 0.9)
        return disc

    return model_with_discrete_bernoulli


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


@pytest.fixture(scope="module", params=[3, 5, 7])
def seed(request):
    return request.param


def test_discrete_sampling_categorical(
    model_with_discrete_categorical, xla_fixture, seed
):
    model = model_with_discrete_categorical()
    trace = pm.sample(
        model=model, sampler_type="compound", xla_fixture=xla_fixture, seed=seed
    )
    round_value = round(
        trace.posterior["model_with_discrete_categorical/disc"].mean().item(), 1
    )
    # check to match the categorical prob parameter
    np.testing.assert_allclose(round_value, 0.9, atol=0.1)


def test_discrete_sampling_bernoulli(model_with_discrete_bernoulli, xla_fixture, seed):
    model = model_with_discrete_bernoulli()
    trace = pm.sample(
        model=model, sampler_type="compound", xla_fixture=xla_fixture, seed=seed
    )
    round_value = round(
        trace.posterior["model_with_discrete_bernoulli/disc"].mean().item(), 1
    )
    # check to match the bernoulli prob parameter
    np.testing.assert_allclose(round_value, 0.9, atol=0.1)


def test_compound_sampling(model_with_discrete_and_continuous, xla_fixture, seed):
    model = model_with_discrete_and_continuous()
    trace = pm.sample(
        model=model, sampler_type="compound", xla_fixture=xla_fixture, seed=seed
    )
    round_value = round(
        trace.posterior["model_with_discrete_and_continuous/disc"].mean().item(), 1
    )
    np.testing.assert_allclose(round_value, 0.9, atol=0.1)
