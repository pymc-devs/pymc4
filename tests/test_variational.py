import pytest
import pymc4 as pm
import numpy as np

from .fixtures.fixtures_models import  simple_model2, bivariate_gaussian


# fmt: off
_test_kwargs = {
    "ADVI": {
        "method": pm.MeanField, 
        "fit_kwargs": {}
    }
}


@pytest.fixture(scope="function", params=list(_test_kwargs), ids=str)
def approximation(request):
    return request.param


def test_fit(approximation, simple_model2):
    model = simple_model2["model"]()
    approx = _test_kwargs[approximation]
    advi = pm.fit(method=approx["method"](model), **approx["fit_kwargs"])
    assert advi is not None
    assert advi.losses.numpy().shape == (approx["fit_kwargs"].get("num_steps") or 10000,)

    q_samples = advi.approximation.sample(10000)
    estimated_mean = simple_model2["estimated_mean"]
    np.testing.assert_allclose(
        np.mean(np.squeeze(q_samples.posterior["model/mu"].values, axis=0)),
        estimated_mean,
        rtol=0.05,
    )


def test_bivariate_shapes(bivariate_gaussian):
    advi = pm.fit(bivariate_gaussian(), num_steps=5000)
    assert advi.losses.numpy().shape == (5000, 2)

    samples = advi.approximation.sample(5000)
    assert samples.posterior["bivariate_gaussian/density"].values.shape == (1, 5000, 2)
