"""
Tests for PyMC4 random variables
"""
from collections import defaultdict, ChainMap
import pytest
import numpy as np
import tensorflow as tf

import pymc4 as pm

_expected_log_prob = defaultdict(lambda: defaultdict(lambda: None))
_check_broadcast = {
    "Flat": {
        "batch_stack": (1, 2),
        "event_stack": (3, 4),
        "samples": [tf.zeros(1), tf.zeros((1, 3, 4)), tf.zeros((1, 5, 3, 4))],
    },
    "HalfFlat": {
        "batch_stack": (1, 2),
        "event_stack": (3, 4),
        "samples": [tf.zeros(1), tf.zeros((1, 3, 4)), tf.zeros((1, 5, 3, 4))],
    },
}
_distribution_conditions = {
    "AR": {
        "scalar_parameters": {
            "num_timesteps": 3,
            "coefficients": [0.1],
            "level_scale": 1,
            "sample": np.array([[0.0], [0.1], [0.2]], dtype="float32"),
        },
        "multidim_parameters": {
            "num_timesteps": 3,
            "coefficients": np.array([[0.1], [-0.8]], dtype="float32"),
            "level_scale": np.ones((2), dtype="float32"),
            "sample": np.zeros((2, 3, 1), dtype="float32"),
        },
    },
    "Bernoulli": {
        "scalar_parameters": {"probs": 0.5, "sample": 1.0},
        "multidim_parameters": {
            "probs": np.array([0.25, 0.5, 0.75], dtype="float32"),
            "sample": np.array([0.0, 0.0, 1.0], dtype="float32"),
        },
    },
    "Beta": {
        "scalar_parameters": {"concentration0": 1.0, "concentration1": 1.0},
        "multidim_parameters": {
            "concentration0": np.array([5.0, 2.0, 1.0], dtype="float32"),
            "concentration1": np.array([1.0, 2.0, 1.0], dtype="float32"),
        },
    },
    "Binomial": {
        "scalar_parameters": {"total_count": 5.0, "probs": 0.5, "sample": 1.0},
        "multidim_parameters": {
            "total_count": np.array([20, 2, 5.0], dtype="float32"),
            "probs": np.array([0.1, 0.7, 0.5], dtype="float32"),
            "sample": np.array([2.0, 2.0, 1.0], dtype="float32"),
        },
    },
    "BetaBinomial": {
        "scalar_parameters": {
            "total_count": 5,
            "concentration0": 1.0,
            "concentration1": 2.0,
            "sample": 1.0,
        },
        "multidim_parameters": {
            "total_count": np.array([2, 5, 10], dtype=np.float32),
            "concentration0": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "concentration1": np.array([3.0, 2.0, 1.0], dtype=np.float32),
            "sample": np.array([1.0, 3.0, 9.0], dtype=np.float32),
        },
    },
    "Categorical": {
        "scalar_parameters": {"probs": np.array([0.1, 0.5, 0.4], dtype="float32"), "sample": 2,},
        "multidim_parameters": {
            "probs": np.array([[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]], dtype="float32"),
            "sample": np.array([2, 2], dtype="int32"),
        },
    },
    "Cauchy": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "Chi2": {
        "scalar_parameters": {"df": 2.0},
        "multidim_parameters": {"df": np.array([4.0, 3.0, 2.0], dtype="float32")},
    },
    "Dirichlet": {
        "scalar_parameters": {
            "concentration": np.array([1.0, 2.0], dtype="float32"),
            "sample": np.array([0.5, 0.5], dtype="float32"),
        },
        "multidim_parameters": {
            "concentration": np.array([[1.0, 2.0], [1.0, 2.0]], dtype="float32"),
            "sample": np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"),
        },
    },
    "DiscreteUniform": {
        "scalar_parameters": {"low": 2.0, "high": 10.0, "sample": 5.0},
        "multidim_parameters": None,
        # DiscreteUniform is derived from FiniteDiscrete, which can only have 1-D outcome tensors
    },
    "Exponential": {
        "scalar_parameters": {"rate": 1.0},
        "multidim_parameters": {"rate": np.array([4.0, 10.0, 1.0], dtype="float32")},
    },
    "Gamma": {
        "scalar_parameters": {"concentration": 3.0, "rate": 2.0},
        "multidim_parameters": {
            "concentration": np.array([2.0, 6.0, 3.0], dtype="float32"),
            "rate": np.array([1.0, 2.0, 2.0], dtype="float32"),
        },
    },
    "GeneralizedNormal": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0, "power": 4.0},
        "multidim_parameters": {
            "loc": np.array([0.25, 0.5], dtype="float32"),
            "scale": np.array([5.0, 10.0], dtype="float32"),
            "power": np.array([1.0, 4.0], dtype="float32"),
        },
    },
    "Geometric": {
        "scalar_parameters": {"probs": 0.5, "sample": 10.0},
        "multidim_parameters": {
            "probs": np.array([0.25, 0.5], dtype="float32"),
            "sample": np.array([5.0, 10.0], dtype="float32"),
        },
    },
    "Gumbel": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "HalfCauchy": {
        "scalar_parameters": {"scale": 1.0},
        "multidim_parameters": {"scale": np.array([1.0, 3.0], dtype="float32")},
    },
    "HalfNormal": {
        "scalar_parameters": {"scale": 3.0},
        "multidim_parameters": {"scale": np.array([6.0, 3.0], dtype="float32")},
    },
    "HalfStudentT": {
        "scalar_parameters": {"scale": 1, "df": 10},
        "multidim_parameters": {
            "scale": np.array([4, 1], dtype="float32"),
            "df": np.array([80, 10], dtype="float32"),
        },
    },
    "InverseGamma": {
        "scalar_parameters": {"concentration": 3, "scale": 2},
        "multidim_parameters": {
            "concentration": np.array([4, 3], dtype="float32"),
            "scale": np.array([2, 2], dtype="float32"),
        },
    },
    "InverseGaussian": {
        "scalar_parameters": {"loc": 1, "concentration": 1},
        "multidim_parameters": {
            "loc": np.array([1, 1], dtype="float32"),
            "concentration": np.array([1, 1], dtype="float32"),
        },
    },
    "Kumaraswamy": {
        "scalar_parameters": {"concentration0": 0.5, "concentration1": 0.5},
        "multidim_parameters": {
            "concentration0": np.array([0.4, 0.5], dtype="float32"),
            "concentration1": np.array([0.4, 0.5], dtype="float32"),
        },
    },
    "LKJ": {
        "scalar_parameters": {
            "dimension": 1,
            "concentration": 1.5,
            "sample": np.array([[1.0]], dtype="float32"),
        },
        "multidim_parameters": {
            "dimension": 1,
            "concentration": np.array([3.0, 1.5], dtype="float32"),
            "sample": np.array([[[1.0]], [[1.0]]], dtype="float32"),
        },
    },
    "LKJCholesky": {
        "scalar_parameters": {
            "dimension": 1,
            "concentration": 1.5,
            "sample": np.array([[1.0]], dtype="float32"),
        },
        "multidim_parameters": {
            "dimension": 1,
            "concentration": np.array([3.0, 1.5], dtype="float32"),
            "sample": np.array([[[1.0]], [[1.0]]], dtype="float32"),
        },
    },
    "Laplace": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "LogNormal": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "Logistic": {
        "scalar_parameters": {"loc": 0.0, "scale": 3.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([3.0, 3.0], dtype="float32"),
        },
    },
    "Moyal": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "Multinomial": {
        "scalar_parameters": {
            "total_count": 4,
            "probs": np.array([0.2, 0.3, 0.5], dtype="float32"),
            "sample": np.array([1.0, 1.0, 2.0], dtype="float32"),
        },
        "multidim_parameters": {
            "total_count": np.array([8, 4], dtype="float32"),
            "probs": np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], dtype="float32"),
            "sample": np.array([[3.0, 1.0, 4.0], [1.0, 1.0, 2.0]], dtype="float32"),
        },
    },
    "LogitNormal": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0},
        "multidim_parameters": {
            "loc": np.array([1.0, 0.0], dtype="float32"),
            "scale": np.array([2.0, 1.0], dtype="float32"),
        },
    },
    "MvNormal": {
        "scalar_parameters": {
            "loc": np.array([1.0, 2.0], dtype="float32"),
            "covariance_matrix": np.array([[0.36, 0.12], [0.12, 0.36]], dtype="float32"),
            "sample": np.array([1.0, 2.0], dtype="float32"),
        },
        "multidim_parameters": {
            "loc": np.array([[1.0, 2.0], [2.0, 3.0]], dtype="float32"),
            "covariance_matrix": np.array(
                [[[0.36, 0.12], [0.12, 0.36]], [[0.36, 0.12], [0.12, 0.36]]], dtype="float32",
            ),
            "sample": np.array([[1.0, 2.0], [2.0, 3.0]], dtype="float32"),
        },
    },
    "MvNormalCholesky": {
        "scalar_parameters": {
            "loc": np.array([1.0, 2.0], dtype="float32"),
            "scale_tril": np.array([[1.0, 0.0], [0.5, 0.866025]], dtype="float32"),
            "sample": np.array([1.0, 2.0], dtype="float32"),
        },
        "multidim_parameters": {
            "loc": np.array([[1.0, 2.0], [2.0, 3.0]], dtype="float32"),
            "scale_tril": np.array(
                [[[1.0, 0.0], [0.5, 0.866025]], [[1.0, 0.0], [0.5, 0.866025]]], dtype="float32",
            ),
            "sample": np.array([[1.0, 2.0], [2.0, 3.0]], dtype="float32"),
        },
    },
    "NegativeBinomial": {
        "scalar_parameters": {"total_count": 3, "probs": 0.6, "sample": 5.0},
        "multidim_parameters": {
            "total_count": np.array([3, 4], dtype="float32"),
            "probs": np.array([0.2, 0.6], dtype="float32"),
            "sample": np.array([2.0, 5.0], dtype="float32"),
        },
    },
    "Normal": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "OrderedLogistic": {
        "scalar_parameters": {
            "loc": 1.0,
            "cutpoints": np.array([0.1, 0.5, 1.0], dtype="float32"),
            "sample": 2,
        },
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "cutpoints": np.array([[0.0, 1.0], [1.0, 2.0]], dtype="float32"),
            "sample": np.array([2, 2], dtype="int32"),
        },
    },
    "Pareto": {
        "scalar_parameters": {"concentration": 1.0, "scale": 0.1, "sample": 5.0},
        "multidim_parameters": {
            "concentration": np.array([1.0, 1.0], dtype="float32"),
            "scale": np.array([0.1, 0.1], dtype="float32"),
            "sample": np.array([5.0, 5.0], dtype="float32"),
        },
    },
    "Poisson": {
        "scalar_parameters": {"rate": 2.0},
        "multidim_parameters": {"rate": np.array([2.0, 3.0], dtype="float32")},
    },
    "StudentT": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0, "df": 10.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1, 1.0], dtype="float32"),
            "df": np.array([10.0, 10.0], dtype="float32"),
        },
    },
    "Triangular": {
        "scalar_parameters": {"low": 0.0, "high": 1.0, "peak": 0.5},
        "multidim_parameters": {
            "low": np.array([0.0, 0.0], dtype="float32"),
            "high": np.array([1.0, 1.0], dtype="float32"),
            "peak": np.array([0.5, 0.5], dtype="float32"),
        },
    },
    "Uniform": {
        "scalar_parameters": {"low": 0.0, "high": 1.0},
        "multidim_parameters": {
            "low": np.array([0.0, -10.0], dtype="float32"),
            "high": np.array([1.0, 10.0], dtype="float32"),
        },
    },
    "VonMises": {
        "scalar_parameters": {"loc": 0.0, "concentration": 1.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 1.0], dtype="float32"),
            "concentration": np.array([1.0, 2.0], dtype="float32"),
        },
    },
    "VonMisesFisher": {
        "scalar_parameters": {
            "mean_direction": np.array([0.0, 1.0], dtype="float32"),
            "concentration": 1.0,
            "sample": np.array([0.0, 1.0], dtype="float32"),
        },
        "multidim_parameters": {
            "mean_direction": np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
            "concentration": np.array([1.0, 1.0], dtype="float32"),
            "sample": np.array([[1.0, 1.0], [0.0, 1.0]], dtype="float32"),
        },
    },
    "Weibull": {
        "scalar_parameters": {"concentration": 1.0, "scale": 1.0},
        "multidim_parameters": {
            "concentration": np.array([1.0, 0.5], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
        },
    },
    "Wishart": {
        "scalar_parameters": {
            "df": 3,
            "scale": np.array([[1]], dtype="float32"),
            "sample": np.array([[1.0]], dtype="float32"),
        },
        "multidim_parameters": {
            "df": np.array([3, 5], dtype="float32"),
            "scale": np.array([[[3]], [[1]]], dtype="float32"),
            "sample": np.array([[[1.0]], [[1.0]]], dtype="float32"),
        },
    },
    "ZeroInflatedBinomial": {
        "scalar_parameters": {"psi": 0.2, "n": 10.0, "p": 0.5, "sample": 0.0},
        "multidim_parameters": {
            "psi": np.array([0.2, 0.2], dtype="float32"),
            "n": np.array([10.0, 10.0], dtype="float32"),
            "p": np.array([0.5, 0.25], dtype="float32"),
            "sample": np.array([0.0, 0.0], dtype="float32"),
        },
    },
    "ZeroInflatedNegativeBinomial": {
        "scalar_parameters": {"psi": 0.2, "mu": 10.0, "alpha": 3.0, "sample": 0.0},
        "multidim_parameters": {
            "psi": np.array([0.2, 0.2], dtype="float32"),
            "mu": np.array([10.0, 10.0], dtype="float32"),
            "alpha": np.array([3.0, 3.0], dtype="float32"),
            "sample": np.array([0.0, 0.0], dtype="float32"),
        },
    },
    "ZeroInflatedPoisson": {
        "scalar_parameters": {"psi": 0.2, "theta": 2.0, "sample": 0},
        "multidim_parameters": {
            "psi": np.array([0.2, 0.2], dtype="float32"),
            "theta": np.array([2.0, 2.0], dtype="float32"),
            "sample": np.array([0.0, 0.0], dtype="float32"),
        },
    },
    "Zipf": {
        "scalar_parameters": {"power": 2.0},
        "multidim_parameters": {"power": np.array([3, 2.0], dtype="float32")},
    },
    "Flat": {
        "scalar_parameters": {"sample": -2.0, "expected": 0.0},
        "multidim_parameters": {
            "sample": np.array([[[-2.0], [-1.0], [0.0], [1.0], [2.0]]]),
            "expected": np.array([[[0.0], [0.0], [0.0], [0.0], [0.0]]]),
        },
    },
    "HalfFlat": {
        "scalar_parameters": {"sample": -2.0, "expected": -np.inf},
        "multidim_parameters": {
            "sample": np.array([[[-2.0], [-1.0], [0.0], [1.0], [2.0]]]),
            "expected": np.array([[[-np.inf], [-np.inf], [-np.inf], [0.0], [0.0]]]),
        },
    },
}

unsupported_dtype_distributions = [
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedBinomial",
    "Poisson",
    "NegativeBinomial",
    "Multinomial",
    "Geometric",
    "DiscreteUniform",
    "Categorical",
    "BetaBinomial",
    "Binomial",
]

_distribution_extra_parameters = {}
for distribution in _distribution_conditions:
    extra_parameters = {
        "dtype": None,
        "validate_args": {"validate_args": True},
        "allow_nan_stats": {"allow_nan_stats": True},
    }
    if (
        not issubclass(getattr(pm, distribution, None), pm.ContinuousDistribution)
        and distribution not in unsupported_dtype_distributions
    ):
        extra_parameters["dtype"] = {"dtype": "int32"}
    if distribution == "AR":
        # time series cannot be configured at the moment
        extra_parameters = {
            "dtype": None,
            "validate_args": None,
            "allow_nan_stats": None,
        }
    _distribution_extra_parameters[distribution] = extra_parameters


@pytest.fixture(scope="function", params=list(_distribution_conditions), ids=str)
def distribution(request):
    return request.param


@pytest.fixture(
    scope="function", params=["scalar_parameters", "multidim_parameters"], ids=str,
)
def distribution_conditions(distribution, request):
    conditions = _distribution_conditions[distribution][request.param]
    if conditions is None:
        pytest.skip("Distribution does not support {}".format(request.param))
    else:
        conditions = conditions.copy()
    log_prob_test_sample = conditions.pop("sample", 0.1)
    expected_log_prob = conditions.pop("expected", None)
    return distribution, conditions, log_prob_test_sample, expected_log_prob


@pytest.fixture(scope="function", params=["dtype", "validate_args", "allow_nan_stats"], ids=str)
def distribution_extra_parameters(distribution, request):
    conditions = _distribution_conditions[distribution]["scalar_parameters"]
    return (
        distribution,
        request.param,
        conditions,
        _distribution_extra_parameters[distribution][request.param],
    )


@pytest.fixture(scope="function", params=list(_check_broadcast), ids=str)
def broadcast_distribution(request):
    return request.param


@pytest.fixture(scope="function", ids=str)
def check_broadcast(broadcast_distribution, request):
    conditions = _check_broadcast[broadcast_distribution]
    batch_stack = conditions.pop("batch_stack", (1, 2))
    event_stack = conditions.pop("event_stack", (3, 4))
    samples = conditions.pop("samples")
    dist_class = getattr(pm, broadcast_distribution)
    dist = dist_class(name=broadcast_distribution, batch_stack=batch_stack, event_stack=event_stack)
    return dist, samples


def test_rvs_logp_and_forward_sample(tf_seed, distribution_conditions):
    """Test forward sampling and evaluating the logp for all random variables."""
    distribution_name, conditions, sample, expected_value = distribution_conditions

    dist = getattr(pm, distribution_name)
    vals = dist(name=distribution_name, **conditions).log_prob(sample)

    assert vals is not None

    if expected_value is not None:
        np.testing.assert_allclose(expected_value, vals, atol=0.01, rtol=0)


def test_rvs_test_point_are_valid(tf_seed, distribution_conditions):
    distribution_name, conditions, sample, expected_value = distribution_conditions

    dist_class = getattr(pm, distribution_name)
    dist = dist_class(name=distribution_name, **conditions)
    test_value = dist.test_value
    if distribution_name in ["Flat", "HalfFlat"]:
        # pytest.skip("Flat and HalfFlat distributions don't support sampling.")
        assert test_value.shape == dist.batch_shape + dist.event_shape
        return
    test_sample = dist.sample()
    logp = dist.log_prob(test_value).numpy()
    assert test_value.shape == test_sample.shape
    assert tuple(test_value.shape.as_list()) == tuple(
        (dist.batch_shape + dist.event_shape).as_list()
    )
    assert not (np.any(np.isinf(logp)) or np.any(np.isnan(logp)))


def test_flat_halfflat_broadcast(tf_seed, check_broadcast):
    """Test the error messages returned by Flat and HalfFlat
    distributions for inconsistent sample shapes"""
    dist, samples = check_broadcast
    for sample in samples:
        with pytest.raises(ValueError, match=r"not consistent"):
            dist.log_prob(sample)


def test_extra_parameters(tf_seed, distribution_extra_parameters):
    (distribution_name, arg_name, conditions, extra_parameters,) = distribution_extra_parameters
    if extra_parameters is None:
        pytest.skip(
            f"Distribution '{distribution_name}' does not support configurable '{arg_name}'"
        )
    dist_class = getattr(pm, distribution_name)
    dist = dist_class(name=distribution_name, **ChainMap(conditions, extra_parameters))
    assert getattr(dist, arg_name) == extra_parameters[arg_name]
    if distribution_name not in ["Flat", "HalfFlat"]:
        # Test that a sample can be drawn using the alternative extra parameters values
        dist.sample()
