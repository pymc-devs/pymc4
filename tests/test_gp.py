import tensorflow as tf
import numpy as np
import pymc4 as pm
from pymc4.gp.util import stabilize
import pytest

# Tensor shapes on which the GP model will be tested
BATCH_AND_FEATURE_SHAPES = [(1,), (2,), (2, 2,)]
SAMPLE_SHAPE = [(1,), (3,)]

# Test all the mean functions in pm.gp module
MEAN_FUNCS = {
    "Zero": {},
    "Constant": {"coef": 5.0},
}

# Test all the covariance functions in pm.gp module
COV_FUNCS = {
    "ExpQuad": {"amplitude": 1.0, "length_scale": 1.0},
}

# Test all the GP models only using a particular
# mean and covariance functions but varying tensor shapes
# NOTE: the mean and covariance functions used here
# must be present in `MEAN_FUNCS` and `COV_FUNCS` resp.
GP_MODELS = {"LatentGP": {"mean_fn": "Zero", "cov_fn": "ExpQuad"}}


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


@pytest.fixture(scope="module", params=list(MEAN_FUNCS), ids=str)
def get_mean_func(request):
    return request.param


@pytest.fixture(scope="module", params=list(COV_FUNCS), ids=str)
def get_cov_func(request):
    return request.param


@pytest.fixture(scope="module", params=list(GP_MODELS), ids=str)
def get_gp_model(request):
    return request.param


def make_func(test_dict, test_func, feature_shape, mod):
    """Returns a mean function from specified name present in test_dict"""
    kwargs = test_dict[test_func]
    func_class = getattr(mod, test_func)
    func = func_class(feature_ndims=len(feature_shape), **kwargs)
    return func


def make_model(data, test_model, feature_shape):
    """Returns a GP model for testing."""
    _, _, feature_shape, _ = data
    gp_params = GP_MODELS[test_model]
    mean_name = gp_params.pop("mean_fn", "Zero")
    cov_name = gp_params.pop("cov_fn", "ExpQuad")
    mean_fn = make_func(MEAN_FUNCS, mean_name, feature_shape, pm.gp.mean)
    cov_fn = make_func(COV_FUNCS, cov_name, feature_shape, pm.gp.cov)
    gp_class = getattr(pm.gp, test_model)
    gp_model = gp_class(mean_fn=mean_fn, cov_fn=cov_fn, **gp_params)
    return gp_model


def test_mean_funcs(tf_seed, get_data, get_mean_func):
    """Test the mean functions present in MEAN_FUNCS dictionary"""
    batch_shape, sample_shape, feature_shape, X = get_data
    mean_func = make_func(MEAN_FUNCS, get_mean_func, feature_shape, pm.gp.mean)
    mean = mean_func(X)
    assert mean is not None
    assert mean.shape == batch_shape + sample_shape


def test_cov_funcs(tf_seed, get_data, get_cov_func):
    """Test the covariance functions present in COV_FUNCS dictionary"""
    batch_shape, sample_shape, feature_shape, X = get_data
    cov_func = make_func(COV_FUNCS, get_cov_func, feature_shape, pm.gp.cov)
    cov = stabilize(cov_func(X, X))
    kernel_point_evaluation = cov_func.evaluate_kernel(X, X)
    assert cov is not None
    assert kernel_point_evaluation is not None
    assert cov.shape.as_list() == list(batch_shape) + list(sample_shape + sample_shape)
    assert np.all(np.linalg.eigvals(cov.numpy()) > 0)


def test_gp_models_prior(tf_seed, get_data, get_gp_model):
    """Test the prior method of a GP mode, if present"""
    batch_shape, sample_shape, feature_shape, X = get_data
    gp_model = make_model(get_data, get_gp_model, feature_shape)
    try:
        prior_dist = gp_model.prior("prior", X)
    except NotImplementedError:
        pytest.skip("Skipping: prior not implemented")

    assert prior_dist is not None
    if sample_shape == (1,):
        assert prior_dist.sample(1).shape == (1,) + batch_shape
    else:
        assert prior_dist.sample(1).shape == (1,) + batch_shape + sample_shape


def test_gp_models_conditional(tf_seed, get_data, get_gp_model):
    """Test the conditional method of a GP mode, if present"""
    batch_shape, sample_shape, feature_shape, X = get_data
    gp_model = make_model(get_data, get_gp_model, feature_shape)
    X_new = tf.random.normal(batch_shape + sample_shape + feature_shape)
    try:
        f = gp_model.prior("f", X).sample(1)[0]
        cond_dist = gp_model.conditional("fcond", X_new, given={"X": X, "f": f})
        cond_samples = cond_dist.sample(3)
    except NotImplementedError:
        pytest.skip("Skipping: conditional not implemented")

    assert cond_samples is not None
    if sample_shape == (1,):
        assert cond_samples.shape == (3,) + batch_shape
    else:
        assert cond_samples.shape == (3,) + batch_shape + sample_shape


def test_covariance_combination(tf_seed, get_cov_func):
    """Test if the combination of various covariance functions
    yield consistent results
    """
    batch_shape, sample_shape, feature_shape, X = (2,), (2,), (2,), tf.random.normal((2, 2, 2))
    kernel1 = make_func(COV_FUNCS, get_cov_func, feature_shape, pm.gp.cov)
    kernel2 = make_func(COV_FUNCS, get_cov_func, feature_shape, pm.gp.cov)
    kernel_add = kernel1 + kernel2
    kernel_mul = kernel1 * kernel2
    cov_add = kernel_add(X, X)
    cov_mul = kernel_mul(X, X)
    assert cov_add is not None
    assert cov_add.shape == batch_shape + sample_shape + sample_shape
    assert np.all(np.linalg.eigvals(cov_add.numpy()) > 0)
    assert cov_mul is not None
    assert cov_mul.shape == batch_shape + sample_shape + sample_shape
    assert np.all(np.linalg.eigvals(cov_mul.numpy()) > 0)


def test_covariance_non_covaraiance_combination(tf_seed, get_cov_func):
    """Test combination of a covariance function with a scalar, vector,
    and broadcastable vector"""
    batch_shape, sample_shape, feature_shape, X = (2,), (3,), (4,), tf.random.normal((2, 3, 4))
    kernel1 = make_func(COV_FUNCS, get_cov_func, feature_shape, pm.gp.cov)
    others = [2.0, np.random.randn(2, 3, 3)]
    for other in others:
        other = np.random.randn(2, 3, 3)
        rmul_kernel = other * kernel1
        radd_kernel = other + kernel1
        eval_rmul = rmul_kernel(X, X)
        eval_radd = radd_kernel(X, X)
        assert rmul_kernel is not None
        assert radd_kernel is not None
        assert eval_rmul.shape == batch_shape + sample_shape + sample_shape
        assert eval_radd.shape == batch_shape + sample_shape + sample_shape


def test_mean_combination(tf_seed, get_mean_func):
    """Test if the combination of various mean functions
    yield consistent results
    """
    batch_shape, sample_shape, feature_shape, X = (2,), (2,), (2,), tf.random.normal((2, 2, 2))
    mean1 = make_func(MEAN_FUNCS, get_mean_func, feature_shape, pm.gp.mean)
    mean2 = make_func(MEAN_FUNCS, get_mean_func, feature_shape, pm.gp.mean)
    mean_add = mean1 + mean2
    mean_mul = mean1 * mean2
    mean_add_val = mean_add(X)
    mean_mul_val = mean_mul(X)
    assert mean_add_val is not None
    assert mean_add_val.shape == batch_shape + sample_shape
    assert mean_mul_val is not None
    assert mean_mul_val.shape == batch_shape + sample_shape


def test_invalid_feature_ndims(tf_seed):
    """Test if an error is throw for inconsistent feature_ndims"""
    with pytest.raises(ValueError, match=r"Cannot combine kernels"):
        kernel1 = pm.gp.cov.ExpQuad(1.0, 1.0, 1)
        kernel2 = pm.gp.cov.ExpQuad(1.0, 1.0, 2)
        kernel = kernel1 + kernel2
    with pytest.raises(ValueError, match=r"Cannot combine means"):
        mean1 = pm.gp.mean.Zero(1)
        mean2 = pm.gp.mean.Zero(2)
        mean = mean1 + mean2
    with pytest.raises(ValueError, match=r"Cannot combine means"):
        mean1 = pm.gp.mean.Zero(1)
        mean2 = pm.gp.mean.Zero(2)
        mean = mean1 * mean2
    with pytest.raises(
        ValueError, match=r"The feature_ndims of mean and covariance functions should be equal"
    ):
        mean = pm.gp.mean.Zero(1)
        cov = pm.gp.cov.ExpQuad(1.0, 1.0, 2)
        gp = pm.gp.LatentGP(mean, cov)


def test_gp_invalid_prior(tf_seed):
    """Test if an error is thrown for invalid model prior"""

    @pm.model
    def invalid_model(gp, X, X_new):
        f = gp.prior("f", X)
        cond = yield gp.conditional("fcond", X_new, given={"X": X, "f": f})

    with pytest.raises(ValueError, match=r"must be a numpy array or tensor"):
        gp = pm.gp.LatentGP(cov_fn=pm.gp.cov.ExpQuad(1.0, 1.0))
        X = tf.random.normal((2, 5, 1))
        X_new = tf.random.normal((2, 2, 1))
        trace = pm.sample(invalid_model(gp, X, X_new), num_samples=1, burn_in=1, num_chains=1)


def test_exp_quad_ls_amplitude(tf_seed):
    """Test the property methods on Exponentiated Quadratic kernel"""
    cov = pm.gp.cov.ExpQuad(1.0, 1.0, 1)
    assert cov.amplitude is not None
    assert cov.length_scale is not None
