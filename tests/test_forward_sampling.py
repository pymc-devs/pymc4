import pytest
import numpy as np
import pymc4 as pm
from pymc4 import forward_sampling


@pytest.fixture(scope="module", params=[(), (1,), (2,), (1, 1), (7, 3)], ids=str)
def sample_shape_fixture(request):
    return request.param


@pytest.fixture(scope="module", params=["NoResampleObserved", "ResampleObserveds"], ids=str)
def sample_from_observed_fixture(request):
    return request.param == "ResampleObserveds"


def test_sample_prior_predictive(sample_shape_fixture, sample_from_observed_fixture):
    observed = np.ones(10)
    @pm.model
    def model():
        sd = yield pm.HalfNormal("sd", 1.)
        x = yield pm.Normal("x", 0, sd, observed=observed)
        y = yield pm.Normal("y", x, 1e-9)

    prior = forward_sampling.sample_prior_predictive(
        model(), sample_shape_fixture, sample_from_observed_fixture
    )
    if sample_from_observed_fixture:
        assert set(["model/sd", "model/x", "model/y"]) == set(prior)
        assert all((value.shape == sample_shape_fixture for value in prior.values()))
    else:
        assert set(["model/sd", "model/y"]) == set(prior)
        assert prior["model/sd"].shape == sample_shape_fixture
        assert prior["model/y"].shape == sample_shape_fixture + observed.shape
        assert np.allclose(prior["model/y"], observed, rtol=1e-5)