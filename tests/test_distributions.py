"""
Tests for PyMC4 random variables
"""
from collections import defaultdict
import pytest
import numpy as np

import pymc4 as pm


_expected_log_prob = defaultdict(lambda: defaultdict(lambda: None))
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
        "scalar_parameters": {"probs": np.array([0.1, 0.5, 0.4], dtype="float32"), "sample": 2},
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
        "multidim_parameters": None,  # DiscreteUniform is derived from FiniteDiscrete, which can only have 1-D outcome tensors
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
    "TruncatedNormal": {
        "scalar_parameters": {"loc": 0.0, "scale": 1.0, "low": 0.0, "high": 2.0},
        "multidim_parameters": {
            "loc": np.array([0.0, 0.0], dtype="float32"),
            "scale": np.array([1.0, 1.0], dtype="float32"),
            "low": np.array([0.0, 0.0]),
            "high": np.array([2.0, 2.0]),
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
            "mean_direction": np.array([[1.0, 1.0], [0.0, 1.0]], dtype="float32"),
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
    # "ZeroInflatedBinomial": {
    #     "scalar_parameters": {"psi": 0.2, "total_count": 10, "p": 0.5, "sample": 0.0},
    #     "multidim_parameters": {"psi": np.array([0.2, 0.2], dtype="float32"), "total_count": np.array([10, 10], dtype="float32"), "p": np.array([0.5, 0.25], dtype="float32"), "sample": np.array([0.0, 0.0], dtype="float32")},
    # },
    #
    # "ZeroInflatedNegativeBinomial": {
    #     "scalar_parameters": {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
    #     "multidim_parameters": {"psi": np.array([0.2, 0.2], dtype="float32"), "mu": np.array([10, 10], dtype="float32"), "alpha": np.array([3, 3], dtype="float32"), "sample": np.array([0, 0], dtype="float32")},
    # },
    # "ZeroInflatedPoisson": {
    #     "scalar_parameters": {"psi": 0.2, "theta": 2, "sample": 0},
    #     "multidim_parameters": {"psi": np.array([0.2, 0.2], dtype="float32"), "theta": np.array([2, 2], dtype="float32"), "sample": np.array([0, 0], dtype="float32")},
    # },
    "Zipf": {
        "scalar_parameters": {"power": 2.0},
        "multidim_parameters": {"power": np.array([3, 2.0], dtype="float32")},
    },
}


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


def test_rvs_logp_and_forward_sample(tf_seed, distribution_conditions):
    """Test forward sampling and evaluating the logp for all random variables."""
    distribution_name, conditions, sample, expected_value = distribution_conditions

    dist = getattr(pm, distribution_name)
    vals = dist(name=distribution_name, **conditions).log_prob(sample)

    assert vals is not None

    if expected_value:
        np.testing.assert_allclose(expected_value, vals, atol=0.01, rtol=0)


def test_rvs_test_point_are_valid(tf_seed, distribution_conditions):
    distribution_name, conditions, sample, expected_value = distribution_conditions

    dist_class = getattr(pm, distribution_name)
    dist = dist_class(name=distribution_name, **conditions)
    test_value = dist.test_value
    test_sample = dist.sample()
    logp = dist.log_prob(test_value).numpy()
    assert test_value.shape == test_sample.shape
    assert tuple(test_value.shape.as_list()) == tuple(
        (dist.batch_shape + dist.event_shape).as_list()
    )
    assert not (np.any(np.isinf(logp)) or np.any(np.isnan(logp)))
