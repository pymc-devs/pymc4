import pytest
import pymc4 as pm


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


def test_discrete_sampling_categorical(model_with_discrete_categorical, xla_fixture):
    model = model_with_discrete_categorical()
    trace = pm.sample(model=model, sampler_type="compound", xla_fixture=xla_fixture)
    round_value = round(trace.posterior["model_with_discrete_categorical/disc"].mean().item(), 1)
    # check to match the categorical prob parameter
    assert round_value == 0.9


def test_discrete_sampling_bernoulli(model_with_discrete_bernoulli, xla_fixture):
    model = model_with_discrete_bernoulli()
    trace = pm.sample(model=model, sampler_type="compound", xla_fixture=xla_fixture)
    round_value = round(trace.posterior["model_with_discrete_bernoulli/disc"].mean().item(), 1)
    # check to match the bernoulli prob parameter
    assert round_value == 0.9


def test_compound_sampling(model_with_discrete_and_continuous, xla_fixture):
    model = model_with_discrete_and_continuous()
    trace = pm.sample(model=model, sampler_type="compound", xla_fixture=xla_fixture)
    round_value = round(trace.posterior["model_with_discrete_and_continuous/disc"].mean().item(), 1)
    assert round_value == 0.9
