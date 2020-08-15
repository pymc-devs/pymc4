import numpy as np
import tensorflow as tf
import pymc4 as pm
from pymc4.gp.util import stabilize

import pytest

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
    (
        "RatQuad",
        {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "scale_mixture_rate": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Matern12",
        {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.05910575], [0.05910575, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Matern32",
        {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.04397209], [0.04397209, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Matern52",
        {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.03701404], [0.03701404, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Exponential",
        {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.24311673], [0.24311673, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Gibbs",
        {
            "length_scale_fn": (lambda x: tf.ones(x.shape)),
            "test_points": [np.array([[1.0], [3.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.13533528], [0.13533528, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Linear",
        {
            "bias_variance": 1.0,
            "slope_variance": 1.0,
            "shift": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[2.0, 4.0], [4.0, 14.0]], dtype=np.float32),
            "expected_point": np.array([2.0, 14.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Polynomial",
        {
            "bias_variance": 1.0,
            "slope_variance": 1.0,
            "shift": 1.0,
            "exponent": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[2.0, 4.0], [4.0, 14.0]], dtype=np.float32),
            "expected_point": np.array([2.0, 14.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Cosine",
        {
            "length_scale": 1.0,
            "amplitude": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 0.47307032], [0.47307032, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
    (
        "Periodic",
        {
            "length_scale": 1.0,
            "amplitude": 1.0,
            "period": 1.0,
            "test_points": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 2,
            "expected_matrix": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "expected_point": np.array([1.0, 1.0], dtype=np.float32),
            "feature_ndims": 1,
        },
    ),
]


@pytest.fixture(scope="function", params=COV_FUNCS, ids=str)
def get_cov_func(request):
    return request.param


@pytest.fixture(scope="function", params=set(k[0] for k in COV_FUNCS), ids=str)
def get_unique_cov_func(request):
    return request.param


def build_class_and_get_test_points(name, kwargs):
    test_points = kwargs.pop("test_points")
    expected_matrix = kwargs.pop("expected_matrix")
    expected_point = kwargs.pop("expected_point", None)
    feature_ndims = kwargs.pop("feature_ndims", 1)
    KernelClass = getattr(pm.gp.cov, name)
    return test_points, expected_matrix, expected_point, feature_ndims, KernelClass


def test_cov_funcs_matrix_shape_psd(tf_seed, get_data, get_unique_cov_func):
    attr_name = get_unique_cov_func
    batch_shape, sample_shape, feature_shape, X = get_data
    kwargs = dict(COV_FUNCS)[attr_name].copy()
    _, _, _, _, KernelClass = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(**kwargs, feature_ndims=len(feature_shape))
    cov = stabilize(kernel(X, X))
    assert cov.shape == batch_shape + sample_shape + sample_shape
    if not np.all(np.linalg.eigvals(cov.numpy()) > 0):
        pytest.xfail("Covariance matrix is not Positive Semi-Definite.")
    if not tf.reduce_all(np.allclose(cov, tf.linalg.matrix_transpose(cov))):
        pytest.xfail("The covariance matrix is not symetric.")


def test_cov_funcs_point_eval_shape(tf_seed, get_data, get_unique_cov_func):
    attr_name = get_unique_cov_func
    batch_shape, sample_shape, feature_shape, X = get_data
    kwargs = dict(COV_FUNCS)[attr_name].copy()
    _, _, _, _, KernelClass = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(**kwargs, feature_ndims=len(feature_shape))
    try:
        point = kernel.evaluate_kernel(X, X)
    except NotImplementedError:
        pytest.skip("`evaluate_kernel` method not implemeted. skipping...")
    assert point.shape == batch_shape + sample_shape
    assert tf.reduce_all(point >= 0.0)


def test_cov_funcs_matrix_no_ard(get_cov_func):
    """Test the covariance functions present in COV_FUNCS list"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    (
        test_points,
        expected_matrix,
        _,
        feature_ndims,
        KernelClass,
    ) = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(**kwargs, feature_ndims=feature_ndims, ARD=False)
    cov = kernel(*test_points).numpy()
    assert cov.dtype == expected_matrix.dtype
    assert cov.shape == expected_matrix.shape
    assert np.allclose(cov, expected_matrix)


def test_cov_funcs_point_eval_no_ard(get_cov_func):
    """Test the `evaluate_kernel` method of covariance functions"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    (test_points, _, expected_point, feature_ndims, KernelClass,) = build_class_and_get_test_points(
        attr_name, kwargs
    )
    kernel = KernelClass(**kwargs, feature_ndims=feature_ndims, ARD=False)
    try:
        point = kernel.evaluate_kernel(*test_points).numpy()
    except NotImplementedError:
        pytest.skip("`evaluate_kernel` method not implemeted. skipping...")
    assert point.shape == expected_point.shape
    assert point.dtype == expected_point.dtype
    assert np.allclose(point, expected_point)
    assert np.all(point >= 0.0)


def test_cov_combination(get_cov_func):
    """Test if the combination of various covariance functions yield consistent results"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    (
        test_points,
        expected_matrix,
        expected_point,
        feature_ndims,
        KernelClass,
    ) = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(**kwargs, feature_ndims=feature_ndims)
    kernel_add = kernel + kernel
    kernel_mul = kernel * kernel
    cov_add = kernel_add(*test_points).numpy()
    cov_mul = kernel_mul(*test_points).numpy()
    assert np.all(np.linalg.eigvals(stabilize(cov_add).numpy()) > 0)
    assert np.all(np.linalg.eigvals(stabilize(cov_mul).numpy()) > 0)
    assert np.allclose(cov_add, expected_matrix * 2)
    assert np.allclose(cov_mul, expected_matrix ** 2)


def test_cov_non_cov_combination(get_cov_func):
    """Test combination of a covariance function with a scalar, vector, and broadcastable vector"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    (
        test_points,
        expected_matrix,
        expected_point,
        feature_ndims,
        KernelClass,
    ) = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(feature_ndims=feature_ndims, **kwargs)
    others = [2.0, np.array([2.0])]
    for other in others:
        kernel_radd = other * kernel
        kernel_rmul = other + kernel
        cov_radd = kernel_rmul(*test_points).numpy()
        cov_rmul = kernel_radd(*test_points).numpy()
        assert np.allclose(cov_radd, other + expected_matrix)
        assert np.allclose(cov_rmul, other * expected_matrix)


def test_cov_funcs_diag(get_cov_func):
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    (
        test_points,
        expected_matrix,
        expected_point,
        feature_ndims,
        KernelClass,
    ) = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(feature_ndims=feature_ndims, **kwargs)
    diag_cov = kernel(*test_points, diag=True).numpy()
    assert diag_cov.shape == expected_matrix.shape
    assert diag_cov.dtype == expected_matrix.dtype
    assert np.allclose(np.diag(diag_cov), np.diag(expected_matrix))
    assert np.all(np.linalg.eigvals(stabilize(diag_cov).numpy()) > 0)


def test_cov_funcs_active_dims(tf_seed):
    X1 = tf.random.normal((20, 10, 5, 5, 3))
    X2 = tf.random.normal((20, 10, 5, 5, 3))
    feature_ndims = 3
    refkernel = pm.gp.cov.ExpQuad(1.0, 1.0, 3)

    active_dims = [4, 3]
    kernel = pm.gp.cov.ExpQuad(1.0, 1.0, feature_ndims, active_dims)

    cov_matrix = kernel(X1, X2)
    expected_matrix = refkernel(X1[:, :, :4, :3, :], X2[:, :, :4, :3, :])
    assert np.allclose(cov_matrix, expected_matrix)

    active_dims = [4, 3, 1]
    kernel = pm.gp.cov.ExpQuad(1.0, 1.0, feature_ndims, active_dims)
    cov_matrix = kernel(X1, X2)
    expected_matrix = refkernel(X1[:, :, :4, :3, :1], X2[:, :, :4, :3, :1])
    assert np.allclose(cov_matrix, expected_matrix)

    active_dims = [[1, 3], 2, 2]
    kernel = pm.gp.cov.ExpQuad(1.0, 1.0, feature_ndims, active_dims)
    cov_matrix = kernel(X1, X2)
    X1new = tf.stack([X1[:, :, 1, :2, :2], X1[:, :, 3, :2, :2]], axis=2)
    X2new = tf.stack([X2[:, :, 1, :2, :2], X2[:, :, 3, :2, :2]], axis=2)
    expected_matrix = refkernel(X1new, X2new)
    assert np.allclose(cov_matrix, expected_matrix)


def test_cov_funcs_invalid_feature_ndims():
    feature_ndims = 0
    with pytest.raises(ValueError, match=r"expected 'feature_ndims' to be an integer"):
        kernel = pm.gp.cov.ExpQuad(1.0, 1.0, feature_ndims)


def test_cov_funcs_invalid_active_dims():
    feature_ndims = 3
    with pytest.raises(
        ValueError, match=r"active_dims' contain more entries than number of feature dimensions",
    ):
        active_dims = [1, 2, 3, 4, 5]
        kernel = pm.gp.cov.ExpQuad(1.0, 1.0, feature_ndims, active_dims)


def test_scaled_cov_kernel_shapes_and_psd(tf_seed, get_data):
    batch_shape, sample_shape, feature_shape, X = get_data
    k = pm.gp.cov.ExpQuad(1.0, feature_ndims=len(feature_shape))
    scal_fn = lambda x: tf.ones(x.shape)
    k_scal = pm.gp.cov.ScaledCov(k, scal_fn)

    cov = k_scal(X, X)
    assert cov.shape == batch_shape + sample_shape + sample_shape
    assert np.all(np.linalg.eigvals(stabilize(cov)) > 0)

    point = k_scal.evaluate_kernel(X, X)
    assert point.shape == batch_shape + sample_shape


def test_scaled_cov():
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    k = pm.gp.cov.ExpQuad(1.0)
    scal_fn = lambda x: tf.ones(x.shape)
    k_scal = pm.gp.cov.ScaledCov(k, scal_fn)

    cov = k_scal(x, x)
    expected = tf.constant([[2.0, 0.03663128], [0.03663128, 2.0]], dtype=np.float32)
    assert np.allclose(cov, expected)


def test_warped_cov_kernel_shapes_and_psd(tf_seed, get_data):
    batch_shape, sample_shape, feature_shape, X = get_data
    k = pm.gp.cov.ExpQuad(1.0, feature_ndims=len(feature_shape))
    warp_fn = lambda x: x[..., :1]
    k_warped = pm.gp.cov.WarpedInput(k, warp_fn)

    cov = k_warped(X, X)
    assert cov.shape == batch_shape + sample_shape + sample_shape
    assert np.all(np.linalg.eigvals(stabilize(cov)) > 0)

    point = k_warped.evaluate_kernel(X, X)
    assert point.shape == batch_shape + sample_shape


def test_warped_cov():
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    k = pm.gp.cov.ExpQuad(1.0)
    warp_fn = lambda x: x[:, :1]
    k_warped = pm.gp.cov.WarpedInput(k, warp_fn)

    cov = k_warped(x, x)
    expected = k(x[:, :1], x[:, :1])
    assert np.allclose(cov, expected)
