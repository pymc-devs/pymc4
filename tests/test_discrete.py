import pytest
import itertools
import pymc4 as pm
import numpy as np
from scipy import stats
import tensorflow as tf


@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        disc = yield pm.Normal("disc", 0, 1)
        return disc

    return simple_model


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


def test_random_walk_sampling(simple_model, xla_fixture):
    model = simple_model()
    trace = pm.sample(
        model=model, sampler_type="randomwalk", num_samples=10, num_chains=4, burn_in=100, step_size=0.1, xla=xla_fixture
    )
