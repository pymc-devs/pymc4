import tensorflow as tf
import numpy as np
import pymc4 as pm
import pytest

OK = True

BATCH_AND_FEATURE_SHAPES = [(1, ), (2, ), (2, 2, )]
SAMPLE_SHAPE = [(1, ), (5, )]

_check_mean = {
    "Zero": {},
    "Constant": {"coef": 5.0},
}

_check_cov = {
    "ExpQuad": {"amplitude": 1.0, "length_scale": 1.0},
}

_check_gp_model = {
    "LatentGP": {}
}


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
    X = np.random.randn(*(get_batch_shape + get_sample_shape + get_feature_shape)).astype(np.float32)
    return get_batch_shape, get_sample_shape, get_feature_shape, X


@pytest.fixture(scope="module", params=list(_check_mean), ids=str)
def get_mean_func(request):
    return request.param


@pytest.fixture(scope="module", params=list(_check_cov), ids=str)
def get_cov_func(request):
    return request.param


@pytest.fixture(scope="module", params=list(_check_gp_model), ids=str)
def get_gp_model(request):
    return request.param


@pytest.fixture(scope="module", ids=str)
def get_prior_model():
    @pm.model
    def prior_model(gp, X):
        f = yield gp.prior('f', X)
        return f
    return prior_model


@pytest.fixture(scope="module", ids=str)
def get_cond_model():
    @pm.model
    def cond_model(gp, X, X_new):
        f = yield gp.prior('f', X)
        fcond = yield gp.conditional('fcond', X_new, given={'X': X, 'f': f})
        return fcond
    return cond_model


def test_mean_funcs(get_data, get_mean_func):
    batch_shape, samples_shape, feature_shape, X = get_data
    mean_args = _check_mean[get_mean_func]
    mean_func_class = getattr(pm.gp.mean, get_mean_func)
    mean_func = mean_func_class(feature_ndims=len(feature_shape), **mean_args)
    mean = mean_func(X)
    assert mean is not None
    assert mean.shape == batch_shape + samples_shape


def test_cov_funcs(get_data, get_cov_func):
    batch_shape, samples_shape, feature_shape, X = get_data
    cov_args = _check_cov[get_cov_func]
    cov_func_class = getattr(pm.gp.cov, get_cov_func)
    cov_func = cov_func_class(feature_ndims=len(feature_shape), **cov_args)
    cov = cov_func(X, X)
    assert cov is not None
    assert cov.shape.as_list() == list(batch_shape) + list(samples_shape + samples_shape)
    # assert tf.reduce_all(tf.linalg.eigvals(cov) > 0.)
    assert np.all(np.linalg.eigvals(cov.numpy()) > 0)


def test_gp_models_prior(get_data, get_mean_func, get_cov_func, get_gp_model, get_prior_model):
    batch_shape, samples_shape, feature_shape, X = get_data
    mean_args = _check_mean[get_mean_func]
    mean_func_class = getattr(pm.gp.mean, get_mean_func)
    mean_func = mean_func_class(feature_ndims=len(feature_shape), **mean_args)
    cov_args = _check_cov[get_cov_func]
    cov_func_class = getattr(pm.gp.cov, get_cov_func)
    cov_func = cov_func_class(feature_ndims=len(feature_shape), **cov_args)
    gp_class = getattr(pm.gp.gp, get_gp_model)
    gp_model = gp_class(mean_fn=mean_func, cov_fn=cov_func)
    try:
        prior_dist = gp_model.prior("prior", X)
    except NotImplementedError:
        pytest.skip("Skipping: prior not implemented")
    
    assert prior_dist is not None
    if samples_shape == (1, ):
        assert prior_dist.sample(1).shape == (1, ) + batch_shape
    else:
        assert prior_dist.sample(1).shape == (1, ) + batch_shape + samples_shape
        