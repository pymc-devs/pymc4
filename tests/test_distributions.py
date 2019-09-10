"""
Tests for PyMC4 random variables
"""
import pytest
import numpy as np

import pymc4 as pm


def random_variable_args():
    """Provide arguments for each random variable."""

    # Commented out tests are currently failing and will be fixed
    _random_variable_args = (
        # ("Bernoulli", {"p": 0.5, "sample": 1.0}),
        ("Beta", {"alpha": 1, "beta": 1}),
        # ("Binomial", {"n": 5.0, "p": 0.5, "sample": 1.0}),
        # ("Categorical", {"p": [0.1, 0.5, 0.4], "sample": 2.0}),
        ("Cauchy", {"alpha": 0, "beta": 1}),
        ("ChiSquared", {"nu": 2}),
        # ("Constant", {"value": 3}),
        # ("Dirichlet", {"a": [1, 2], "sample": [0.5, 0.5]}),
        # ("DiscreteUniform", {"lower": 2.0, "upper": 10.0, "sample": 5.0}),
        ("Exponential", {"lam": 1}),
        ("Gamma", {"alpha": 3.0, "beta": 2.0}),
        # ("Geometric", {"p": 0.5, "sample": 10.0}),
        ("Gumbel", {"mu": 0, "beta": 1}),
        ("HalfCauchy", {"beta": 1}),
        ("HalfNormal", {"sigma": 3.0}),
        # ("HalfStudentT", {"sigma": 1, "nu": 10}),
        ("InverseGamma", {"alpha": 3, "beta": 2}),
        ("InverseGaussian", {"mu": 1, "lam": 1}),
        ("Kumaraswamy", {"a": 0.5, "b": 0.5}),
        # ("LKJ", {"n": 1, "eta": 1.5, "sample": [[1.0]]}),
        ("Laplace", {"mu": 0, "b": 1}),
        ("LogNormal", {"mu": 0, "sigma": 1}),
        ("Logistic", {"mu": 0, "s": 3}),
        # ("Multinomial", {"n": 4, "p": [0.2, 0.3, 0.5], "sample": [1.0, 1.0, 2.0]}),
        ("LogitNormal", {"mu": 0, "sigma": 1}),
        # (
        #     "MvNormal",
        #     {"mu": [1, 2], "cov": [[0.36, 0.12], [0.12, 0.36]], "sample": [1.0, 2.0]},
        # ),
        # ("NegativeBinomial", {"mu": 3, "alpha": 6, "sample": 5.0}),
        ("Normal", {"mu": 0, "sigma": 1}),
        ("Pareto", {"alpha": 1, "m": 0.1, "sample": 5.0}),
        # ("Poisson", {"mu": 2}),
        ("StudentT", {"mu": 0, "sigma": 1, "nu": 10}),
        ("Triangular", {"lower": 0.0, "upper": 1.0, "c": 0.5}),
        ("Uniform", {"lower": 0, "upper": 1}),
        ("VonMises", {"mu": 0, "kappa": 1}),
        # ("VonMisesFisher", {"mu": [0, 1], "kappa": 1, "sample": [0.0, 1.0]}),
        # ("Weibull", {"beta": 0.1, "alpha": 1.0}),
        # ("Wishart", {"nu": 3, "V": [[1]], "sample": [[1.0]]}),
        # ("ZeroInflatedBinomial", {"psi": 0.2, "n": 10, "p": 0.5, "sample": 0.0}),
        # (
        #     "ZeroInflatedNegativeBinomial",
        #     {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
        # ),
        # ("ZeroInflatedPoisson", {"psi": 0.2, "theta": 2, "sample": 0}),
        # ("Zipf", {"alpha": 2.0}),
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
    x = pm.Normal("NormA", mu=0, sigma=1)
    y = pm.Normal("NormB", mu=1, sigma=2)

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
