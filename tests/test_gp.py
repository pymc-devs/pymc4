import tensorflow as tf
import numpy as np
import pymc4 as pm
import pytest

BATCH_AND_FEATURE_SHAPES = [(1,), (2,), (2, 2,)]
SAMPLE_SHAPE = [(1,), (5,)]

_check_mean = {
    "Zero": {},
    "Constant": {"coef": 5.0},
}

_check_cov = {
    "ExpQuad": {"amplitude": 1.0, "length_scale": 1.0},
}

_check_gp_model = {"LatentGP": {}}


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
    @pm.model(name="prior_model")
    def prior_model(gp, X):
        f = yield gp.prior("f", X)
        return f

    return prior_model


@pytest.fixture(scope="module", ids=str)
def get_cond_model():
    @pm.model(name="cond_model")
    def cond_model(gp, X, X_new):
        f = yield gp.prior("f", X)
        fcond = yield gp.conditional("fcond", X_new, given={"X": X, "f": f})
        return fcond

    return cond_model


def get_func(test_dict, test_func, feature_shape, mod):
    kwargs = test_dict[test_func]
    func_class = getattr(mod, test_func)
    func = func_class(feature_ndims=len(feature_shape), **kwargs)
    return func


def test_mean_funcs(tf_seed, get_data, get_mean_func):
    batch_shape, sample_shape, feature_shape, X = get_data
    mean_func = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    mean = mean_func(X)
    assert mean is not None
    assert mean.shape == batch_shape + sample_shape


def test_cov_funcs(tf_seed, get_data, get_cov_func):
    batch_shape, sample_shape, feature_shape, X = get_data
    cov_func = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    cov = cov_func(X, X)
    assert cov is not None
    assert cov.shape.as_list() == list(batch_shape) + list(sample_shape + sample_shape)
    # assert tf.reduce_all(tf.linalg.eigvals(cov) > 0.)
    assert np.all(np.linalg.eigvals(cov.numpy()) > 0)


def test_gp_models_prior(tf_seed, get_data, get_mean_func, get_cov_func, get_gp_model):
    batch_shape, sample_shape, feature_shape, X = get_data
    mean_func = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    cov_func = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    gp_class = getattr(pm.gp.gp, get_gp_model)
    gp_model = gp_class(mean_fn=mean_func, cov_fn=cov_func)
    try:
        prior_dist = gp_model.prior("prior", X)
        # trace = pm.sample(get_prior_model(gp_model, X), num_samples=3, num_chains=1)
    except NotImplementedError:
        pytest.skip("Skipping: prior not implemented")

    assert prior_dist is not None
    # assert trace.posterior['prior_model/f'] is not None
    if sample_shape == (1,):
        assert prior_dist.sample(1).shape == (1,) + batch_shape
        # assert trace.posterior['prior_model/f'].shape == (1, 3, ) + batch_shape
    else:
        assert prior_dist.sample(1).shape == (1,) + batch_shape + sample_shape
        # assert trace.posterior['prior_model/f'].shape == (1, 3, ) + batch_shape + sample_shape


def test_gp_models_conditional(
    tf_seed, get_data, get_mean_func, get_cov_func, get_gp_model, get_cond_model
):
    batch_shape, sample_shape, feature_shape, X = get_data
    X_new = tf.random.normal(batch_shape + sample_shape + feature_shape)
    mean_func = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    cov_func = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    gp_class = getattr(pm.gp.gp, get_gp_model)
    gp_model = gp_class(mean_fn=mean_func, cov_fn=cov_func)
    try:
        trace = pm.sample(get_cond_model(gp_model, X, X_new), num_samples=3, num_chains=1)
        # f = gp_model.prior('f', X).sample(3)
        # cond_dist = gp_model.conditional('fcond', X_new, given={'X': X, 'f': f})
        # cond_samples = cond_dist.sample(1)
    except NotImplementedError:
        pytest.skip("Skipping: conditional not implemented")

    # assert cond_samples is not None
    assert trace.posterior["cond_model/fcond"] is not None
    if sample_shape == (1,):
        # assert cond_samples.shape == (1, 3, ) + batch_shape
        assert trace.posterior["cond_model/fcond"].shape == (1, 3,) + batch_shape
    else:
        # assert cond_samples.shape == (1, 3, ) + batch_shape + sample_shape
        assert trace.posterior["cond_model/fcond"].shape == (1, 3,) + batch_shape + sample_shape


def test_covariance_combination(tf_seed, get_data, get_cov_func):
    batch_shape, sample_shape, feature_shape, X = get_data
    kernel1 = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    kernel2 = get_func(_check_cov, get_cov_func, feature_shape, pm.gp.cov)
    kernel_add = kernel1 + kernel2
    kernel_mul = kernel1 * kernel2
    cov_add = kernel_add(X, X)
    cov_mul = kernel_mul(X, X)
    assert cov_add is not None
    assert cov_add.shape.as_list() == list(batch_shape) + list(sample_shape + sample_shape)
    # assert tf.reduce_all(tf.linalg.eigvals(cov_add) > 0.)
    assert np.all(np.linalg.eigvals(cov_add.numpy()) > 0)
    assert cov_mul is not None
    assert cov_mul.shape.as_list() == list(batch_shape) + list(sample_shape + sample_shape)
    # assert tf.reduce_all(tf.linalg.eigvals(cov_mul) > 0.)
    assert np.all(np.linalg.eigvals(cov_mul.numpy()) > 0)


def test_mean_combination(tf_seed, get_data, get_mean_func):
    batch_shape, sample_shape, feature_shape, X = get_data
    mean1 = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    mean2 = get_func(_check_mean, get_mean_func, feature_shape, pm.gp.mean)
    mean_add = mean1 + mean2
    mean_mul = mean1 * mean2
    mean_add_val = mean_add(X)
    mean_cov_val = mean_mul(X)
    assert mean_add_val is not None
    assert mean_add_val.shape == batch_shape + sample_shape
    assert mean_cov_val is not None
    assert mean_cov_val.shape == batch_shape + sample_shape


def test_invalid_feature_ndims(tf_seed):
    with pytest.raises(ValueError, match=r"Cannot combine kernels"):
        kernel1 = pm.gp.cov.ExpQuad(1.0, 1.0, 1)
        kernel2 = pm.gp.cov.ExpQuad(1.0, 1.0, 2)
        kernel = kernel1 + kernel2
    with pytest.raises(ValueError, match=r"Cannot combine means"):
        mean1 = pm.gp.mean.Zero(1)
        mean2 = pm.gp.mean.Zero(2)
        kernel = mean1 + mean2
    with pytest.raises(
        ValueError, match=r"The feature_ndims of mean and covariance functions should be the same"
    ):
        mean = pm.gp.mean.Zero(1)
        cov = pm.gp.cov.ExpQuad(1.0, 1.0, 2)
        gp = pm.gp.gp.LatentGP(mean, cov)


def test_exp_quad_invalid_amplitude(tf_seed):
    with pytest.raises(ValueError, match=r"must not contains zero or negative entries"):
        kernel = pm.gp.cov.ExpQuad(-1.0, 1.0, 1)


def test_gp_invalid_prior(tf_seed):
    @pm.model
    def invalid_model(gp, X, X_new):
        f = gp.prior("f", X)
        cond = yield gp.conditional("fcond", X_new, given={"X": X, "f": f})

    with pytest.raises(ValueError, match=r"must be a numpy array or tensor"):
        gp = pm.gp.gp.LatentGP()
        X = tf.random.normal((2, 5, 1))
        X_new = tf.random.normal((2, 2, 1))
        trace = pm.sample(invalid_model(gp, X, X_new), num_samples=1, burn_in=1, num_chains=1)
