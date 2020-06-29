"""
Fixtures for tests_gp
"""
import pytest
import numpy as np
import tensorflow as tf

# Test all the GP models only using a particular
# mean and covariance functions but varying tensor shapes
# NOTE: the mean and covariance functions used here
# must be present in `MEAN_FUNCS` and `COV_FUNCS` resp.

GP_MODELS = [
    (
        "LatentGP",
        {"mean_fn": ("Zero", {}), "cov_fn": ("ExpQuad", {"amplitude": 1.0, "length_scale": 1.0})},
    ),
]

@pytest.fixture(scope="module", params=GP_MODELS, ids=str)
def get_gp_model(request):
    return request.param

# Test all the mean functions in pm.gp module
MEAN_FUNCS = [
    (
        "Zero",
        {
            "test_point": np.array([[1.0], [2.0]], dtype=np.float64),
            "expected": np.array([0.0, 0.0], dtype=np.float64),
            "feature_ndims": 1,
        },
    ),
    (
        "Constant",
        {
            "coef": 5.0,
            "test_point": np.array([[1.0], [2.0]], dtype=np.float64),
            "expected": np.array([5.0, 5.0], dtype=np.float64),
            "feature_ndims": 1,
        },
    ),
]

# Tensor shapes on which the GP model will be tested
BATCH_AND_FEATURE_SHAPES = [(1,), (2,), (2, 2,)]
SAMPLE_SHAPE = [(1,), (3,)]

@pytest.fixture(scope="module", params=MEAN_FUNCS, ids=str)
def get_mean_func(request):
    return request.param


# Test all the covariance functions in pm.gp module
COV_FUNCS = [
    (
        "ExpQuad",
        {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.01831564], [0.01831564, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Constant",
        {
            "coef": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "WhiteNoise",
        {
            "noise": 1e-4,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1e-4, 0.0], [0.0, 1e-4]], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
]

@pytest.fixture(scope="function", params=COV_FUNCS, ids=str)
def get_cov_func(request):
    return request.param

# TODO - this is a bit hacky
@pytest.fixture(scope="function", ids=str)
def get_all_cov_func(request):
    return COV_FUNCS


@pytest.fixture(scope="function", params=set(k[0] for k in COV_FUNCS), ids=str)
def get_unique_cov_func(request):
    return request.param

@pytest.fixture(scope="module", params=BATCH_AND_FEATURE_SHAPES, ids=str)
def get_batch_shape(request):
    return request.param


@pytest.fixture(scope="module", params=SAMPLE_SHAPE, ids=str)
def get_sample_shape(request):
    return request.param


@pytest.fixture(scope="module", params=BATCH_AND_FEATURE_SHAPES, ids=str)
def get_feature_shape(request):
    return request.param


@pytest.fixture(scope="module")
def get_data(get_batch_shape, get_sample_shape, get_feature_shape):
    X = tf.random.normal(get_batch_shape + get_sample_shape + get_feature_shape)
    return get_batch_shape, get_sample_shape, get_feature_shape, X