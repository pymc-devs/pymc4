import pytest
import pymc4 as pm
import numpy as np


@pytest.fixture(scope="function")
def normal_model():
    mean, var = 3, 5
    data = np.random.normal(mean, var, size=200)

    @pm.model
    def normal_model():
        mu = yield pm.Normal("mu", 0, 10)
        sigma = yield pm.Exponential("sigma", 1)
        likelihood = yield pm.Normal("ll", mu, sigma, observed=data)
        return likelihood

    return mean, var, normal_model


@pytest.fixture(scope="function")
def bivariate_gaussian():
    mu = np.zeros(2, dtype=np.float32)
    cov = np.array([[1, 0.8], [0.8, 1]], dtype=np.float32)

    @pm.model
    def bivariate_gaussian():
        density = yield pm.MvNormal("density", loc=mu, covariance_matrix=cov)
        return density

    return bivariate_gaussian


def test_fit(normal_model):

    mean, var, normal_model = normal_model
    mean_field = pm.MeanField(normal_model())
    track = {param.name: param for param in mean_field.trainable_variables}

    def trace_fn(traceable_quantities):
        return {"loss": traceable_quantities.loss, **track}

    advi = pm.fit(method=mean_field, num_steps=40000, trace_fn=trace_fn)
    assert advi is not None

    samples = advi.approximation.sample(10000)
    free_rvs = ["normal_model/mu", "normal_model/__log_sigma", "normal_model/sigma"]
    for rv in free_rvs:
        assert samples.posterior[rv].values.shape == (1, 10000)
    for rv_track in advi.losses:
        assert advi.losses[rv_track].numpy().shape == (40000,)
    np.testing.assert_allclose(mean, mean_field.approx.mean()[0], atol=0.5)
    np.testing.assert_allclose(var, np.exp(mean_field.approx.mean()[1]), atol=0.5)


def test_bivariate_shapes(bivariate_gaussian):
    advi = pm.fit(bivariate_gaussian())
    assert advi.losses.numpy().shape == (10000, 2)

    samples = advi.approximation.sample(5000)
    assert samples.posterior["bivariate_gaussian/density"].values.shape == (1, 5000, 2)
