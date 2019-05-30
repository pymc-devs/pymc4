"""
Tests for PyMC4 random variables
"""
import pytest
import numpy as np


from .. import distributions
from .._model import model


def random_variable_args():
    """Provide arguments for each random variable."""
    _random_variable_args = (
        (distributions.Bernoulli, {"p": 0.5, "sample": 1.0}),
        (distributions.Beta, {"alpha": 1, "beta": 1}),
        (distributions.Binomial, {"n": 5.0, "p": 0.5, "sample": 1.0}),
        (distributions.Categorical, {"p": [0.1, 0.5, 0.4], "sample": 2.0}),
        (distributions.Cauchy, {"alpha": 0, "beta": 1}),
        (distributions.ChiSquared, {"nu": 2}),
        (distributions.Constant, {"value": 3}),
        (distributions.Dirichlet, {"a": [1, 2], "sample": [0.5, 0.5]}),
        (distributions.DiscreteUniform, {"lower": 2.0, "upper": 10.0, "sample": 5.0}),
        (distributions.Exponential, {"lam": 1}),
        (distributions.Gamma, {"alpha": 3.0, "beta": 2.0}),
        (distributions.Geometric, {"p": 0.5, "sample": 10.0}),
        (distributions.Gumbel, {"mu": 0, "beta": 1}),
        (distributions.HalfCauchy, {"beta": 1}),
        (distributions.HalfNormal, {"sigma": 3.0}),
        (distributions.HalfStudentT, {"sigma": 1, "nu": 10}),
        (distributions.InverseGamma, {"alpha": 3, "beta": 2}),
        (distributions.InverseGaussian, {"mu": 1, "lam": 1}),
        (distributions.Kumaraswamy, {"a": 0.5, "b": 0.5}),
        (distributions.LKJ, {"n": 1, "eta": 1.5, "sample": [[1.0]]}),
        (distributions.Laplace, {"mu": 0, "b": 1}),
        (distributions.LogNormal, {"mu": 0, "sigma": 1}),
        (distributions.Logistic, {"mu": 0, "s": 3}),
        (distributions.Multinomial, {"n": 4, "p": [0.2, 0.3, 0.5], "sample": [1.0, 1.0, 2.0]}),
        (distributions.LogitNormal, {"mu": 0, "sigma": 1}),
        (
            distributions.MvNormal,
            {"mu": [1, 2], "cov": [[0.36, 0.12], [0.12, 0.36]], "sample": [1.0, 2.0]},
        ),
        (distributions.NegativeBinomial, {"mu": 3, "alpha": 6, "sample": 5.0}),
        (distributions.Normal, {"mu": 0, "sigma": 1}),
        (distributions.Pareto, {"alpha": 1, "m": 0.1, "sample": 5.0}),
        (distributions.Poisson, {"mu": 2}),
        (distributions.StudentT, {"mu": 0, "sigma": 1, "nu": 10}),
        (distributions.Triangular, {"lower": 0.0, "upper": 1.0, "c": 0.5}),
        (distributions.Uniform, {"lower": 0, "upper": 1}),
        (distributions.VonMises, {"mu": 0, "kappa": 1}),
        (distributions.VonMisesFisher, {"mu": [0, 1], "kappa": 1, "sample": [0.0, 1.0]}),
        (distributions.Weibull, {"beta": 0.1, "alpha": 1.0}),
        (distributions.Wishart, {"nu": 3, "V": [[1]], "sample": [[1.0]]}),
        (distributions.ZeroInflatedBinomial, {"psi": 0.2, "n": 10, "p": 0.5, "sample": 0.0}),
        (
            distributions.ZeroInflatedNegativeBinomial,
            {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
        ),
        (distributions.ZeroInflatedPoisson, {"psi": 0.2, "theta": 2, "sample": 0}),
        (distributions.Zipf, {"alpha": 2}),
    )

    ids = [dist[0].__name__ for dist in _random_variable_args]
    return {
        "argnames": ("randomvariable", "kwargs"),
        "argvalues": _random_variable_args,
        "ids": ids,
    }


@pytest.mark.parametrize(**random_variable_args())
def test_rvs_logp_and_forward_sample(tf_seed, randomvariable, kwargs, request):
    """Test forward sampling and evaluating the logp for all random variables."""
    sample = kwargs.pop("sample", 0.1)
    expected_value = kwargs.pop("expected", None)

    # Logps can only be evaluated in a model
    @model
    def test_model():
        randomvariable(name=request.node.name, **kwargs, validate_args=True)

    # TODO: Fix tests args for that use tfd.Mixture in their implementation
    if not any(
        [
            (randomvariable is rv)
            for rv in (
                distributions.ZeroInflatedBinomial,
                distributions.ZeroInflatedNegativeBinomial,
                distributions.ZeroInflatedPoisson,
            )
        ]
    ):

        test_model = test_model.configure()
        log_prob = test_model.make_log_prob_function()
        # Assert that values are returned with no exceptions
        vals = log_prob(sample)

        assert vals is not None

        if expected_value:
            np.testing.assert_allclose(expected_value, vals, atol=0.01, rtol=0)

    else:
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/mixture.py # L119

        with pytest.raises(TypeError) as err:
            test_model = test_model.configure()
            # _ = log_prob(sample)
            assert (
                "cat must be a Categorical distribution,"
                " but saw: tfp.distributions.TransformedDistribution"
            ) == str(err)


def test_rvs_backend_arithmetic(tf_seed):
    """Test backend arithmetic implemented by the `WithBackendArithmetic` class."""
    x = distributions.Normal(mu=0, sigma=1)
    y = distributions.Normal(mu=1, sigma=2)

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
