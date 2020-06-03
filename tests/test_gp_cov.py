import numpy as np

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
    assert np.all(np.linalg.eigvals(cov.numpy()) > 0)


def test_funcs_point_eval_shape(tf_seed, get_data, get_unique_cov_func):
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


def test_cov_funcs_matrix(get_cov_func):
    """Test the covariance functions present in COV_FUNCS list"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    test_points, expected_matrix, _, feature_ndims, KernelClass = build_class_and_get_test_points(
        attr_name, kwargs
    )
    kernel = KernelClass(**kwargs, feature_ndims=feature_ndims)
    cov = kernel(*test_points).numpy()
    assert cov.dtype == expected_matrix.dtype
    assert cov.shape == expected_matrix.shape
    assert np.allclose(cov, expected_matrix)


def test_cov_funcs_point_eval(get_cov_func):
    """Test the `evaluate_kernel` method of covariance functions"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    test_points, _, expected_point, feature_ndims, KernelClass = build_class_and_get_test_points(
        attr_name, kwargs
    )
    kernel = KernelClass(**kwargs, feature_ndims=feature_ndims)
    try:
        point = kernel.evaluate_kernel(*test_points).numpy()
    except NotImplementedError:
        pytest.skip("`evaluate_kernel` method not implemeted. skipping...")
    assert point.shape == expected_point.shape
    assert point.dtype == expected_point.dtype
    assert np.allclose(point, expected_point)
    assert np.all(point >= 0.0)


def test_covariance_combination(get_cov_func):
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


def test_covariance_non_covaraiance_combination(get_cov_func):
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
