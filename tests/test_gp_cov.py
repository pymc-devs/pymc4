import numpy as np
import tensorflow as tf
import pymc4 as pm
from pymc4.gp.util import stabilize
import pytest

from .fixtures.fixtures_gp import get_data, get_batch_shape, get_sample_shape, get_feature_shape
from .fixtures.fixtures_gp import get_gp_model, get_mean_func, get_cov_func, get_unique_cov_func, get_all_cov_func

def build_class_and_get_test_points(name, kwargs):
    test_points = kwargs.pop("test_points")
    expected_matrix = kwargs.pop("expected_matrix")
    expected_point = kwargs.pop("expected_point", None)
    feature_ndims = kwargs.pop("feature_ndims", 1)
    KernelClass = getattr(pm.gp.cov, name)
    return test_points, expected_matrix, expected_point, feature_ndims, KernelClass


def test_cov_funcs_matrix_shape_psd(tf_seed, get_data, get_all_cov_func, get_unique_cov_func):
    attr_name = get_unique_cov_func
    batch_shape, sample_shape, feature_shape, X = get_data
    COV_FUNCS = get_all_cov_func
    kwargs = dict(COV_FUNCS)[attr_name].copy()
    _, _, _, _, KernelClass = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(**kwargs, feature_ndims=len(feature_shape))
    cov = stabilize(kernel(X, X))
    assert cov.shape == batch_shape + sample_shape + sample_shape
    assert np.all(np.linalg.eigvals(cov.numpy()) > 0)


def test_cov_funcs_point_eval_shape(tf_seed, get_data, get_cov_func, get_unique_cov_func, get_all_cov_func):
    attr_name = get_unique_cov_func
    batch_shape, sample_shape, feature_shape, X = get_data
    COV_FUNCS = get_all_cov_func
    kwargs = dict(COV_FUNCS)[attr_name].copy()
    _, _, _, _, KernelClass = build_class_and_get_test_points(attr_name, kwargs)
    kernel = KernelClass(**kwargs, feature_ndims=len(feature_shape))
    try:
        point = kernel.evaluate_kernel(X, X)
    except NotImplementedError:
        pytest.skip("`evaluate_kernel` method not implemeted. skipping...")
    assert point.shape == batch_shape + sample_shape


def test_cov_funcs_matrix_no_ard(get_cov_func):
    """Test the covariance functions present in COV_FUNCS list"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    test_points, expected_matrix, _, feature_ndims, KernelClass = build_class_and_get_test_points(
        attr_name, kwargs
    )
    kernel = KernelClass(**kwargs, feature_ndims=feature_ndims, ARD=False)
    cov = kernel(*test_points).numpy()
    assert cov.dtype == expected_matrix.dtype
    assert cov.shape == expected_matrix.shape
    assert np.allclose(cov, expected_matrix)


def test_cov_funcs_point_eval_no_ard(get_cov_func):
    """Test the `evaluate_kernel` method of covariance functions"""
    attr_name = get_cov_func[0]
    kwargs = get_cov_func[1].copy()
    test_points, _, expected_point, feature_ndims, KernelClass = build_class_and_get_test_points(
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
        ValueError, match=r"active_dims' contain more entries than number of feature dimensions"
    ):
        active_dims = [1, 2, 3, 4, 5]
        kernel = pm.gp.cov.ExpQuad(1.0, 1.0, feature_ndims, active_dims)
