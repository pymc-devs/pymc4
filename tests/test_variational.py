import pytest
import pymc4 as pm
import numpy as np


@pytest.fixture(scope="function")
def simple_model():
    unknown_mean = -5
    known_sigma = 3
    data_points = 1000
    data = np.random.normal(unknown_mean, known_sigma, size=data_points)
    prior_mean = 4
    prior_sigma = 2

    # References - http://patricklam.org/teaching/conjugacy_print.pdf
    precision = 1 / prior_sigma ** 2 + data_points / known_sigma ** 2
    estimated_mean = (
        prior_mean / prior_sigma ** 2 + (data_points * np.mean(data) / known_sigma ** 2)
    ) / precision

    @pm.model
    def model():
        mu = yield pm.Normal("mu", prior_mean, prior_sigma)
        ll = yield pm.Normal("ll", mu, known_sigma, observed=data)
        return ll

    return dict(data_points=data_points, data=data, estimated_mean=estimated_mean, model=model)


# fmt: off
_test_kwargs = {
    "ADVI": {
        "method": pm.MeanField, 
        "fit_kwargs": {}
    },
    "FullRank ADVI": {
        "method": pm.FullRank,
        "fit_kwargs": {}
    }
}


@pytest.fixture(scope="function", params=list(_test_kwargs), ids=str)
def approximation(request):
    return request.param


def test_fit(approximation, simple_model):
    model = simple_model["model"]()
    approx = _test_kwargs[approximation]
    advi = pm.fit(method=approx["method"](model), **approx["fit_kwargs"])
    assert advi is not None
    assert advi.losses.numpy().shape == (approx["fit_kwargs"].get("num_steps") or 10000,)

    q_samples = advi.approximation.sample(10000)
    estimated_mean = simple_model["estimated_mean"]
    np.testing.assert_allclose(
        np.mean(np.squeeze(q_samples.posterior["model/mu"].values, axis=0)),
        estimated_mean,
        rtol=0.05,
    )


@pytest.fixture(scope="function")
def bivariate_gaussian():
    mu = np.zeros(2, dtype=np.float32)
    cov = np.array([[1, 0.8], [0.8, 1]], dtype=np.float32)

    @pm.model
    def bivariate_gaussian():
        density = yield pm.MvNormal("density", loc=mu, covariance_matrix=cov)
        return density

    return bivariate_gaussian


def test_bivariate_shapes(bivariate_gaussian):
    advi = pm.fit(bivariate_gaussian(), num_steps=5000)
    assert advi.losses.numpy().shape == (5000, )

    samples = advi.approximation.sample(5000)
    assert samples.posterior["bivariate_gaussian/density"].values.shape == (1, 5000, 2)
