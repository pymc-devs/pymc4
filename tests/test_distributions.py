"""
Tests for PyMC4 random variables
"""
import pytest
import numpy as np
from numpy import testing as npt

import pymc4 as pm
import tensorflow as tf


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
        # ("HalfStudentT", {"scale": 1, "nu": 10}),
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


@pytest.mark.parametrize(**random_variable_args())
def test_rvs_logp_and_forward_sample(tf_seed, distribution_name, kwargs):
    """Test forward sampling and evaluating the logp for all random variables."""
    sample = kwargs.pop("sample", 0.1)
    expected_value = kwargs.pop("expected", None)

    dist = getattr(pm, distribution_name)
    vals = dist(name=distribution_name, **kwargs).log_prob(sample)

    assert vals is not None

    if expected_value:
        np.testing.assert_allclose(expected_value, vals, atol=0.01, rtol=0)


@pytest.mark.xfail(
    raises=TypeError, reason="Raising Typerror at the moment. Should double check if still needed"
)
def test_rvs_backend_arithmetic(tf_seed):
    """Test backend arithmetic implemented by the `WithBackendArithmetic` class."""
    x = pm.Normal("NormA", loc=0, scale=1)
    y = pm.Normal("NormB", loc=1, scale=2)

    assert x + y is not None
    assert x - y is not None
    assert x * y is not None
    # TODO test __matmul__ once random variables support shapes.
    # assert x @ y is not None
    assert x / y is not None
    assert x // y is not None
    assert x % y is not None
    assert x ** y is not None
    assert -x is not None


def test_deterministic_cannot_be_observed():
    @pm.model
    def half_normal(loc, scale):
        normal = yield pm.Normal(name="normal", loc=loc, scale=scale, plate=20)
        yield pm.Deterministic(name="abs_normal", value=tf.math.abs(normal), observed=np.zeros(20))

    with pytest.raises(TypeError) as excinfo:
        pm.evaluate_model(half_normal(0, 1))
    assert str(excinfo.value).startswith("Deterministics cannot have observed values")


def test_deterministic():
    @pm.model
    def half_normal(loc, scale):
        normal = yield pm.Normal(name="normal", loc=loc, scale=scale, plate=20)
        yield pm.Deterministic(name="abs_normal", value=tf.math.abs(normal))

    _, state = pm.evaluate_model(half_normal(0, 1))
    normal = state.untransformed_values["half_normal/normal"].numpy()
    abs_normal = state.untransformed_values["half_normal/abs_normal"].numpy()

    # There were some entries less than 0
    assert np.any(normal < 0)

    # The deterministic variable is all positive
    npt.assert_allclose(np.abs(normal), abs_normal)
