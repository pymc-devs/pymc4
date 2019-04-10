"""
Tests for PyMC4 random variables
"""
import pytest
import numpy as np


from .. import random_variables
from .._model import model


def random_variable_args():
    """Provide arguments for each random variable."""
    _random_variable_args = (
        (random_variables.Bernoulli, {"p": 0.5, "sample":1.0}),
        (random_variables.Beta, {"alpha": 1, "beta": 1}),
        (random_variables.Binomial, {"n": 5.0, "p": 0.5, "sample": 1.0}),
        (random_variables.Categorical, {"p": [0.1, 0.5, 0.4], "sample":2.0}),
        (random_variables.Cauchy, {"alpha": 0, "beta": 1}),
        (random_variables.ChiSquared, {"nu": 2}),
        (random_variables.Constant, {"value": 3}),
        (random_variables.Dirichlet, {"a": [1, 2], "sample": [0.5, 0.5]}),
        (random_variables.DiscreteUniform, {"lower": 2.0, "upper": 10.0, "sample": 5.0}),
        (random_variables.Exponential, {"lam": 1}),
        (random_variables.Gamma, {"alpha": 3.0, "beta": 2.0}),
        (random_variables.Geometric, {"p": 0.5, "sample": 10.0}),
        (random_variables.Gumbel, {"mu": 0, "beta": 1}),
        (random_variables.HalfCauchy, {"beta": 1}),
        (random_variables.HalfNormal, {"sigma": 3.0}),
        (random_variables.HalfStudentT, {"sigma": 1, "nu": 10}),
        (random_variables.InverseGamma, {"alpha": 3, "beta": 2}),
        (random_variables.InverseGaussian, {"mu": 1, "lam": 1}),
        (random_variables.Kumaraswamy, {"a": 0.5, "b": 0.5}),
        (random_variables.LKJ, {"n": 1, "eta": 1.5, "sample": [[1.0]]}),
        (random_variables.Laplace, {"mu": 0, "b": 1}),
        (random_variables.LogNormal, {"mu": 0, "sigma": 1}),
        (random_variables.Logistic, {"mu": 0, "s": 3}),
        (random_variables.Multinomial, {"n": 4, "p": [0.2, 0.3, 0.5], "sample": [1.0, 1.0, 2.0]}),
        (random_variables.LogitNormal, {"mu": 0, "sigma": 1}),
        (
            random_variables.MvNormal,
            {"mu": [1, 2], "cov": [[0.36, 0.12], [0.12, 0.36]], "sample": [1.0, 2.0]},
        ),
        (random_variables.NegativeBinomial, {"mu": 3, "alpha": 6, "sample": 5.0}),
        (random_variables.Normal, {"mu": 0, "sigma": 1}),
        (random_variables.Pareto, {"alpha": 1, "m": 0.1, "sample": 5.0}),
        (random_variables.Poisson, {"mu": 2}),
        (random_variables.StudentT, {"mu": 0, "sigma": 1, "nu": 10}),
        (random_variables.Triangular, {"lower": 0.0, "upper": 1.0, "c": 0.5}),
        (random_variables.Uniform, {"lower": 0, "upper": 1}),
        (random_variables.VonMises, {"mu": 0, "kappa": 1}),
        (random_variables.VonMisesFisher, {"mu": [0, 1], "kappa": 1, "sample": [0.0, 1.0]}),
        (random_variables.Weibull, {"beta": 0.1, "alpha": 1.0}),
        (random_variables.Wishart, {"nu": 3, "V": [[1]], "sample": [[1.0]]}),
        (random_variables.ZeroInflatedBinomial, {"psi": 0.2, "n": 10, "p": 0.5, "sample": 0.0}),
        (
            random_variables.ZeroInflatedNegativeBinomial,
            {"psi": 0.2, "mu": 10, "alpha": 3, "sample": 0},
        ),
        (random_variables.ZeroInflatedPoisson, {"psi": 0.2, "theta": 2, "sample": 0}),
        (random_variables.Zipf, {"alpha": 2}),
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
    if not any([(randomvariable is rv)
            for rv in (random_variables.ZeroInflatedBinomial, random_variables.ZeroInflatedNegativeBinomial,random_variables.ZeroInflatedPoisson)]):

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
            assert ("cat must be a Categorical distribution," 
                   " but saw: tfp.distributions.TransformedDistribution") == str(err)


def test_rvs_backend_arithmetic(tf_seed):
    """Test backend arithmetic implemented by the `WithBackendArithmetic` class."""
    x = random_variables.Normal(mu=0, sigma=1)
    y = random_variables.Normal(mu=1, sigma=2)

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
