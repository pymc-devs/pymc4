import pytest
import pymc4 as pm
import numpy as np
from pymc4 import distributions as dist
import tensorflow as tf


@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        norm = yield pm.Normal("norm", 0, 1)
        return norm

    return simple_model


@pytest.fixture(scope="function")
def simple_model_with_deterministic(simple_model):
    @pm.model()
    def simple_model_with_deterministic():
        norm = yield simple_model()
        determ = yield pm.Deterministic("determ", norm * 2)
        return determ

    return simple_model_with_deterministic


@pytest.fixture(scope="module", params=[True, False], ids=str)
def xla_fixture(request):
    return request.param


def test_sample_deterministics(simple_model_with_deterministic, xla_fixture):
    #    if xla_fixture:
    #        pytest.skip("XLA in sampling is still not fully supported")
    model = simple_model_with_deterministic()
    trace, stats = pm.inference.sampling.sample(
        model=model, num_samples=10, num_chains=4, burn_in=100, step_size=0.1, xla=xla_fixture
    )
    norm = "simple_model_with_deterministic/simple_model/norm"
    determ = "simple_model_with_deterministic/determ"
    np.testing.assert_allclose(trace[determ], trace[norm] * 2)
