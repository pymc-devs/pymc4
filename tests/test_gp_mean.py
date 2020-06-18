import tensorflow as tf
import numpy as np

import pymc4 as pm

import pytest

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


@pytest.fixture(scope="module", params=MEAN_FUNCS, ids=str)
def get_mean_func(request):
    return request.param


def test_mean_funcs(tf_seed, get_data, get_mean_func):
    """Test the mean functions present in MEAN_FUNCS dictionary"""
    # Build the mean function
    attr_name = get_mean_func[0]
    kwargs = get_mean_func[1]
    test_point = kwargs.pop("test_point", None)
    expected = kwargs.pop("expected", None)
    feature_ndims = kwargs.pop("feature_ndims", 1)
    MeanClass = getattr(pm.gp.mean, attr_name)

    # Get data to compute on.
    batch_shape, sample_shape, feature_shape, X = get_data

    # Build and evaluate the mean function.
    mean_func = MeanClass(**kwargs, feature_ndims=len(feature_shape))
    val = mean_func(X)

    # Test 1 : Tensor Shape evaluations
    assert val.shape == batch_shape + sample_shape

    # Test 2 : Point evaluations
    if test_point is not None:
        mean_func = MeanClass(**kwargs, feature_ndims=feature_ndims)
        val = mean_func(test_point).numpy()

        # We need to be careful about the dtypes. Even though tensorflow uses float32
        # default dtype. The function should not break for other dtypes also.
        assert val.dtype == expected.dtype
        assert val.shape == expected.shape
        assert np.allclose(val, expected)


def test_mean_combination(tf_seed, get_mean_func):
    """Test if the combination of various mean functions yield consistent results"""
    # Data to compute on.
    batch_shape, sample_shape, feature_shape, X = (2,), (2,), (2,), tf.random.normal((2, 2, 2))
    attr_name = get_mean_func[0]
    kwargs = get_mean_func[1]
    test_point = kwargs.pop("test_point", None)
    expected = kwargs.pop("expected", None)
    feature_ndims = kwargs.pop("feature_ndims", 1)
    MeanClass = getattr(pm.gp.mean, attr_name)

    # Build and evaluate the mean function.
    mean_func = MeanClass(**kwargs, feature_ndims=len(feature_shape))

    # Get the combinations of the mean functions
    mean_add = mean_func + mean_func
    mean_prod = mean_func * mean_func

    # Evaluate the combinations
    mean_add_val = mean_add(X)
    mean_prod_val = mean_prod(X)

    # Test 1 : Shape evaluations
    assert mean_add_val.shape == batch_shape + sample_shape
    assert mean_prod_val.shape == batch_shape + sample_shape

    # Test 2 : Point evaluations
    if test_point is not None:
        mean_func = MeanClass(**kwargs, feature_ndims=feature_ndims)

        # Get the combinations of the mean functions
        mean_add = mean_func + mean_func
        mean_prod = mean_func * mean_func

        # Evaluate the combinations
        mean_add_val = mean_add(test_point).numpy()
        mean_prod_val = mean_prod(test_point).numpy()

        # We need to be careful about the dtypes. Even though tensorflow uses float32
        # default dtype. The function should not break for other dtypes also.
        assert mean_add_val.dtype == expected.dtype
        assert mean_add_val.shape == expected.shape
        assert mean_prod_val.dtype == expected.dtype
        assert mean_prod_val.shape == expected.shape
        assert np.allclose(mean_add_val, 2 * expected)
        assert np.allclose(mean_prod_val, expected ** 2)
