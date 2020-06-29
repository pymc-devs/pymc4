"""
Fixtures for commonly used test models
"""

import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf
from pymc4 import distributions as dist
import itertools

@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        norm = yield pm.Normal("norm", 0, 1)
        return norm

    return simple_model

# previously called simple_model in test_executor 
# main difference is no @pm.model() and we use dist.Normal rather than pm.Normal
@pytest.fixture(scope="function")
def simple_model_dist():
    def simple_model():
        norm = yield dist.Normal("norm", 0, 1)
        return norm

    return simple_model

@pytest.fixture(scope="module")
def simple_model_class():
    class ClassModel:
        @pm.model(method=True)
        def class_model_method(self):
            norm = yield pm.Normal("norm", 0, 1)
            return norm

    return ClassModel()    

@pytest.fixture(scope="function")
def simple_model_with_deterministic(simple_model):
    @pm.model()
    def simple_model_with_deterministic():
        norm = yield simple_model()
        determ = yield pm.Deterministic("determ", norm * 2)
        return determ

    return simple_model_with_deterministic


@pytest.fixture(scope="function")
def simple_model_no_free_rvs():
    @pm.model()
    def simple_model_no_free_rvs():
        norm = yield pm.Normal("norm", 0, 1, observed=1)
        return norm

    return simple_model_no_free_rvs


# Previously called simple_model in tests_variational
# Suggest this could be called something better?
@pytest.fixture(scope="function")
def simple_model2():
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

    return dict(data_points=data_points, data=data, estimated_mean=estimated_mean, model=model)


@pytest.fixture(
    scope="function",
    params=itertools.product(
        [(), (3,), (3, 2)], [(), (2,), (4,), (5, 4)], [(), (1,), (10,), (10, 10)]
    ),
    ids=str,
)
def unvectorized_model(request):
    norm_shape, observed_shape, batch_size = request.param
    observed = np.ones(observed_shape)

    @pm.model()
    def unvectorized_model():
        norm = yield pm.Normal("norm", 0, 1, batch_stack=norm_shape)
        determ = yield pm.Deterministic("determ", tf.reduce_max(norm))
        output = yield pm.Normal("output", determ, 1, observed=observed)

    return unvectorized_model, norm_shape, observed, batch_size


@pytest.fixture(scope="function")
def bivariate_gaussian():
    mu = np.zeros(2, dtype=np.float32)
    cov = np.array([[1, 0.8], [0.8, 1]], dtype=np.float32)

    @pm.model
    def bivariate_gaussian():
        density = yield pm.MvNormal("density", loc=mu, covariance_matrix=cov)
        return density

    return bivariate_gaussian

# TODO - lots of code duplication with fixtures_executors.deterministics_in_nested_models
@pytest.fixture(scope="function")
def deterministics_in_nested_models():
    @pm.model
    def nested_model(cond):
        x = yield pm.Normal("x", cond, 1)
        dx = yield pm.Deterministic("dx", x + 1)
        return dx

    @pm.model
    def outer_model():
        cond = yield pm.HalfNormal("cond", 1)
        dcond = yield pm.Deterministic("dcond", cond * 2)
        dx = yield nested_model(dcond)
        ddx = yield pm.Deterministic("ddx", dx)
        return ddx

    expected_untransformed = {
        "outer_model",
        "outer_model/cond",
        "outer_model/nested_model",
        "outer_model/nested_model/x",
    }
    expected_transformed = {"outer_model/__log_cond"}
    expected_deterministics = {
        "outer_model/dcond",
        "outer_model/ddx",
        "outer_model/nested_model/dx",
    }
    deterministic_mapping = {
        "outer_model/dcond": (["outer_model/__log_cond"], lambda x: np.exp(x) * 2),
        "outer_model/ddx": (["outer_model/nested_model/dx"], lambda x: x),
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1),
    }

    return (
        outer_model,
        expected_untransformed,
        expected_transformed,
        expected_deterministics,
        deterministic_mapping,
    )
