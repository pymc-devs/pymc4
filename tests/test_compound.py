import pytest
import pymc4 as pm


@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        var1 = yield pm.Normal("var1", 0, 1)
        return var1

    return simple_model


@pytest.fixture(scope="function")
def compound_model():
    @pm.model()
    def compound_model():
        var1 = yield pm.Normal("var1", 0, 1)
        var2 = yield pm.Bernoulli("var2", 0.5)
        return var2

    return compound_model


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


@pytest.fixture(scope="module", params=[3, 5, 7])
def seed(request):
    return request.param


@pytest.fixture(scope="module", params=["hmc", "nuts", "randomwalkm", "compound"])
def sampler_type(request):
    return request.param


def test_samplers_on_compound_model(compound_model, seed, xla_fixture, sampler_type):
    def _execute():
        model = compound_model()
        trace = pm.sample(model, sampler_type=sampler_type, xla_fixture=xla_fixture, seed=seed)
        var1 = round(trace.posterior["compound_model/var1"].mean().item(), 1)
        # int32 dtype variable
        var2 = tf.reduce_sum(trace.posterior["compound_model/var2"]) / (1000 * 10)
        assert var1 == 0.0 and var2 == 0.5

    if compound_model in ["compound", "randomwalkm"]:
        # execute normally if sampler supports discrete distributions
        _execute()
    else:
        with pytest.raises(ValueError):
            # else check for the exception thrown
            _execute()


def test_samplers_on_simple_model(simple_model, seed, xla_fixture, sampler_type):
    model = simple_model()
    trace = pm.sample(model, sampler_type=sampler_type, xla_fixture=xla_fixture, seed=seed)
    var1 = round(trace.posterior["simple_model/var1"].mean().item(), 1)
    assert var1 == 0.0
