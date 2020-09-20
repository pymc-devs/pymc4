import pytest
import pymc4 as pm
import numpy as np
from scipy.stats import norm


@pytest.fixture(scope="function")
def conjugate_normal_model():
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

    return dict(estimated_mean=estimated_mean, known_sigma=known_sigma, data=data, model=model)


# fmt: off
_test_kwargs = {
    "ADVI": {
        "method": pm.MeanField,
        "fit_kwargs": {},
        "sample_kwargs": {"n": 500, "include_log_likelihood": True},
    },
    "FullRank ADVI": {
        "method": pm.FullRank,
        "fit_kwargs": {}
    },
    "FullRank ADVI: sample_size=2": {
        "method": pm.FullRank,
        "fit_kwargs": {"sample_size": 2}
    }
}

# fmt: on
@pytest.fixture(scope="function", params=list(_test_kwargs), ids=str)
def approximation(request):
    return request.param


def test_fit(approximation, conjugate_normal_model):
    model = conjugate_normal_model["model"]()
    approx = _test_kwargs[approximation]
    advi = pm.fit(method=approx["method"](model), **approx["fit_kwargs"])
    assert advi is not None
    assert advi.losses.numpy().shape == (approx["fit_kwargs"].get("num_steps", 10000),)

    q_samples = advi.approximation.sample(**approx.get("sample_kwargs", {"n": 1000}))

    # Calculating mean from all draws and comparing to the actual one
    calculated_mean = q_samples.posterior["model/mu"].mean(dim=("chain", "draw"))
    np.testing.assert_allclose(calculated_mean, conjugate_normal_model["estimated_mean"], rtol=0.05)

    if "sample_kwargs" in approx and approx["sample_kwargs"].get("include_log_likelihood"):
        sample_mean = q_samples.posterior["model/mu"].sel(chain=0, draw=0)  # Single draw
        ll_from_scipy = norm.logpdf(
            conjugate_normal_model["data"], sample_mean, conjugate_normal_model["known_sigma"]
        )
        ll_from_pymc4 = q_samples.log_likelihood["model/ll"].sel(chain=0, draw=0)
        assert ll_from_scipy.shape == ll_from_pymc4.shape
        np.testing.assert_allclose(ll_from_scipy, ll_from_pymc4, rtol=1e-4)


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
    assert advi.losses.numpy().shape == (5000,)

    samples = advi.approximation.sample(5000)
    assert samples.posterior["bivariate_gaussian/density"].values.shape == (1, 5000, 2)


def test_advi_with_deterministics(simple_model_with_deterministic):
    advi = pm.fit(simple_model_with_deterministic(), num_steps=1000)
    samples = advi.approximation.sample(100)
    norm = "simple_model_with_deterministic/simple_model/norm"
    determ = "simple_model_with_deterministic/determ"
    np.testing.assert_allclose(samples.posterior[determ], samples.posterior[norm] * 2)


def test_advi_with_deterministics_in_nested_models(deterministics_in_nested_models):
    (
        model,
        *_,
        deterministic_mapping,
    ) = deterministics_in_nested_models
    advi = pm.fit(model(), num_steps=1000)
    samples = advi.approximation.sample(100)
    for deterministic, (inputs, op) in deterministic_mapping.items():
        np.testing.assert_allclose(
            samples.posterior[deterministic], op(*[samples.posterior[i] for i in inputs]), rtol=1e-6
        )
