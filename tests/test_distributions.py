"""
Tests for PyMC4 random variables
"""
from collections import defaultdict
import pytest
import numpy as np

import pymc4 as pm


_expected_log_prob = defaultdict(lambda: defaultdict(lambda: None))
_distribution_conditions = {
    "Bernoulli": {
        "scalar_parameters": {"probs": 0.5, "sample": 1.0},
        "multidim_parameters": {"probs": [0.25, 0.5, 0.75], "sample": [0.0, 0.0, 1.0]},
    },
    "Beta": {
        "scalar_parameters": {"concentration0": 1, "concentration1": 1},
        "multidim_parameters": {"concentration0": [5, 2, 1], "concentration1": [1, 2, 1]},
    },
    "Binomial": {
        "scalar_parameters": {"total_count": 5.0, "probs": 0.5, "sample": 1.0},
        "multidim_parameters": {
            "total_count": [20, 2, 5.0],
            "probs": [0.1, 0.7, 0.5],
            "sample": [2.0, 2.0, 1.0],
        },
    },
    "Categorical": {
        "scalar_parameters": {"probs": [0.1, 0.5, 0.4], "sample": 2.0},
        "multidim_parameters": {"probs": [[0.1, 0.5, 0.4], [0.1, 0.5, 0.4]], "sample": [2.0, 2.0]},
    },
    "Cauchy": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {"loc": [0, 0], "scale": [1, 1]},
    },
    "Chi2": {"scalar_parameters": {"df": 2}, "multidim_parameters": {"df": [4, 3, 2]},},
    "Dirichlet": {
        "scalar_parameters": {"concentration": [1, 2], "sample": [0.5, 0.5]},
        "multidim_parameters": {
            "concentration": [[1, 2], [1, 2]],
            "sample": [[0.5, 0.5], [0.5, 0.5]],
        },
    },
    "DiscreteUniform": {
        "scalar_parameters": {"low": 2.0, "high": 10.0, "sample": 5.0},
        "multidim_parameters": {"low": [1, 2.0], "high": [3, 10.0], "sample": [2, 5.0]},
    },
    "Exponential": {"scalar_parameters": {"rate": 1}, "multidim_parameters": {"rate": [4, 10, 1]},},
    "Gamma": {
        "scalar_parameters": {"concentration": 3.0, "rate": 2.0},
        "multidim_parameters": {"concentration": [2, 6, 3.0], "rate": [1, 2, 2.0]},
    },
    "Geometric": {
        "scalar_parameters": {"probs": 0.5, "sample": 10.0},
        "multidim_parameters": {"probs": [0.25, 0.5], "sample": [5, 10.0]},
    },
    "Gumbel": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {"loc": [0, 0], "scale": [1, 1]},
    },
    "HalfCauchy": {"scalar_parameters": {"scale": 1}, "multidim_parameters": {"scale": [1, 3]},},
    "HalfNormal": {
        "scalar_parameters": {"scale": 3.0},
        "multidim_parameters": {"scale": [6, 3.0]},
    },
    "HalfStudentT": {
        "scalar_parameters": {"scale": 1, "df": 10},
        "multidim_parameters": {"scale": [4, 1], "df": [80, 10]},
    },
    "InverseGamma": {
        "scalar_parameters": {"concentration": 3, "scale": 2},
        "multidim_parameters": {"concentration": [4, 3], "scale": [2, 2]},
    },
    "InverseGaussian": {
        "scalar_parameters": {"loc": 1, "concentration": 1},
        "multidim_parameters": {"loc": [1, 1], "concentration": [1, 1]},
    },
    "Kumaraswamy": {
        "scalar_parameters": {"concentration0": 0.5, "concentration1": 0.5},
        "multidim_parameters": {"concentration0": [0.4, 0.5], "concentration1": [0.4, 0.5]},
    },
    "LKJ": {
        "scalar_parameters": {"dimension": 1, "concentration": 1.5, "sample": [[1.0]]},
        "multidim_parameters": {
            "dimension": 1,
            "concentration": [3, 1.5],
            "sample": [[[1.0]], [[1.0]]],
        },
    },
    "Laplace": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {"loc": [0, 0], "scale": [1, 1]},
    },
    "LogNormal": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {"loc": [0, 0], "scale": [1, 1]},
    },
    "Logistic": {
        "scalar_parameters": {"loc": 0, "scale": 3},
        "multidim_parameters": {"loc": [0, 0], "scale": [3, 3]},
    },
    "Multinomial": {
        "scalar_parameters": {
            "total_count": 4,
            "probs": [0.2, 0.3, 0.5],
            "sample": [1.0, 1.0, 2.0],
        },
        "multidim_parameters": {
            "total_count": [8, 4],
            "probs": [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
            "sample": [[3.0, 1.0, 4.0], [1.0, 1.0, 2.0]],
        },
    },
    "LogitNormal": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {"loc": [1, 0], "scale": [2, 1]},
    },
    "MvNormal": {
        "scalar_parameters": {
            "loc": [1, 2],
            "covariance_matrix": [[0.36, 0.12], [0.12, 0.36]],
            "sample": [1.0, 2.0],
        },
        "multidim_parameters": {
            "loc": [[1, 2], [2, 3]],
            "covariance_matrix": [[[0.36, 0.12], [0.12, 0.36]], [[0.36, 0.12], [0.12, 0.36]]],
            "sample": [[1.0, 2.0], [2.0, 3.0]],
        },
    },
    "NegativeBinomial": {
        "scalar_parameters": {"total_count": 3, "probs": 0.6, "sample": 5.0},
        "multidim_parameters": {"total_count": [3, 4], "probs": [0.2, 0.6], "sample": [2.0, 5.0]},
    },
    "Normal": {
        "scalar_parameters": {"loc": 0, "scale": 1},
        "multidim_parameters": {"loc": [0, 0], "scale": [1, 1]},
    },
    "Pareto": {
        "scalar_parameters": {"concentration": 1, "scale": 0.1, "sample": 5.0},
        "multidim_parameters": {"concentration": [1, 1], "scale": [0.1, 0.1], "sample": [5.0, 5.0]},
    },
    "Poisson": {"scalar_parameters": {"rate": 2}, "multidim_parameters": {"rate": [2, 3]},},
    "StudentT": {
        "scalar_parameters": {"loc": 0, "scale": 1, "df": 10},
        "multidim_parameters": {"loc": [0, 0], "scale": [1, 1], "df": [10, 10]},
    },
    "Triangular": {
        "scalar_parameters": {"low": 0.0, "high": 1.0, "peak": 0.5},
        "multidim_parameters": {"low": [0.0, 0.0], "high": [1.0, 1.0], "peak": [0.5, 0.5]},
    },
    "Uniform": {
        "scalar_parameters": {"low": 0, "high": 1},
        "multidim_parameters": {"low": [0, -10], "high": [1, 10]},
    },
    "VonMises": {
        "scalar_parameters": {"loc": 0, "concentration": 1},
        "multidim_parameters": {"loc": [0, 1], "concentration": [1, 2]},
    },
    "VonMisesFisher": {
        "scalar_parameters": {"mean_direction": [0, 1], "concentration": 1, "sample": [0.0, 1.0]},
        "multidim_parameters": {
            "mean_direction": [[1, 1], [0, 1]],
            "concentration": [1, 1],
            "sample": [[1, 1], [0.0, 1.0]],
        },
    },
    # "Weibull": {
    #     "scalar_parameters": {"beta": 0.1, "alpha": 1.0},
    #     "multidim_parameters": {"beta": 0.1, "alpha": 1.0},
    # },
    "Wishart": {
        "scalar_parameters": {"df": 3, "scale": [[1]], "sample": [[1.0]]},
        "multidim_parameters": {"df": 3, "scale": [[[3]], [[1]]], "sample": [[[1.0]], [[1.0]]]},
    },
    # "ZeroInflatedBinomial": {
    #     "scalar_parameters": {"psi": 0.2, "total_count": 10, "p": 0.5, "sample": 0.0},
    #     "multidim_parameters": {"psi": 0.2, "total_count": 10, "p": 0.5, "sample": 0.0},
    # },
    #
    # "ZeroInflatedNegativeBinomial": {
    #     "scalar_parameters": {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
    #     "multidim_parameters": {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
    # },
    # "ZeroInflatedPoisson": {
    #     "scalar_parameters": {"psi": 0.2, "theta": 2, "sample": 0},
    #     "multidim_parameters": {"psi": 0.2, "theta": 2, "sample": 0},
    # },
    "Zipf": {"scalar_parameters": {"power": 2.0}, "multidim_parameters": {"power": [3, 2.0]},},
}


def random_variable_args():
    """Provide arguments for each random variable."""

    # Commented out tests are currently failing and will be fixed
    _random_variable_args = (
        ("Bernoulli", {"probs": 0.5, "sample": 1.0}),
        ("Beta", {"concentration0": 1, "concentration1": 1}),
        ("Binomial", {"total_count": 5.0, "probs": 0.5, "sample": 1.0}),
        ("Categorical", {"probs": [0.1, 0.5, 0.4], "sample": 2.0}),
        ("Cauchy", {"loc": 0, "scale": 1}),
        ("Chi2", {"df": 2}),
        ("Dirichlet", {"concentration": [1, 2], "sample": [0.5, 0.5]}),
        ("DiscreteUniform", {"low": 2.0, "high": 10.0, "sample": 5.0}),
        ("Exponential", {"rate": 1}),
        ("Gamma", {"concentration": 3.0, "rate": 2.0}),
        ("Geometric", {"probs": 0.5, "sample": 10.0}),
        ("Gumbel", {"loc": 0, "scale": 1}),
        ("HalfCauchy", {"scale": 1}),
        ("HalfNormal", {"scale": 3.0}),
        ("HalfStudentT", {"scale": 1, "df": 10}),
        ("InverseGamma", {"concentration": 3, "scale": 2}),
        ("InverseGaussian", {"loc": 1, "concentration": 1}),
        ("Kumaraswamy", {"concentration0": 0.5, "concentration1": 0.5}),
        ("LKJ", {"dimension": 1, "concentration": 1.5, "sample": [[1.0]]}),
        ("Laplace", {"loc": 0, "scale": 1}),
        ("LogNormal", {"loc": 0, "scale": 1}),
        ("Logistic", {"loc": 0, "scale": 3}),
        ("Multinomial", {"total_count": 4, "probs": [0.2, 0.3, 0.5], "sample": [1.0, 1.0, 2.0]}),
        ("LogitNormal", {"loc": 0, "scale": 1}),
        (
            "MvNormal",
            {
                "loc": [1, 2],
                "covariance_matrix": [[0.36, 0.12], [0.12, 0.36]],
                "sample": [1.0, 2.0],
            },
        ),
        ("NegativeBinomial", {"total_count": 3, "probs": 0.6, "sample": 5.0}),
        ("Normal", {"loc": 0, "scale": 1}),
        ("Pareto", {"concentration": 1, "scale": 0.1, "sample": 5.0}),
        ("Poisson", {"rate": 2}),
        ("StudentT", {"loc": 0, "scale": 1, "df": 10}),
        ("Triangular", {"low": 0.0, "high": 1.0, "peak": 0.5}),
        ("Uniform", {"low": 0, "high": 1}),
        ("VonMises", {"loc": 0, "concentration": 1}),
        ("VonMisesFisher", {"mean_direction": [0, 1], "concentration": 1, "sample": [0.0, 1.0]}),
        # ("Weibull", {"beta": 0.1, "alpha": 1.0}),
        ("Wishart", {"df": 3, "scale": [[1]], "sample": [[1.0]]}),
        # ("ZeroInflatedBinomial", {"psi": 0.2, "total_count": 10, "p": 0.5, "sample": 0.0}),
        # (
        #     "ZeroInflatedNegativeBinomial",
        #     {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
        # ),
        # ("ZeroInflatedPoisson", {"psi": 0.2, "theta": 2, "sample": 0}),
        ("Zipf", {"power": 2.0}),
    )

    ids = [dist[0] for dist in _random_variable_args]
    return {
        "argnames": ("distribution_name", "kwargs"),
        "argvalues": _random_variable_args,
        "ids": ids,
    }


@pytest.fixture(scope="function", params=list(_distribution_conditions), ids=str)
def distribution(request):
    return request.param


@pytest.fixture(
    scope="function", params=["scalar_parameters", "multidim_parameters"], ids=str,
)
def distribution_conditions(distribution, request):
    conditions = _distribution_conditions[distribution][request.param].copy()
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
    sample = dist.sample()
    print(test_value)
    print(sample)
    print(dist.log_prob(sample))
    logp = dist.log_prob(test_value).numpy()
    assert tuple(test_value.shape.as_list()) == tuple(
        (dist.batch_shape + dist.event_shape).as_list()
    )
    assert not (np.any(np.isinf(logp)) or np.any(np.isnan(logp)))
