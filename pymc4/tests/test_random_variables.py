"""
Tests for PyMC4 random variables
"""
import pytest
from .. import random_variables


def random_variable_args():
    """Provide arguments for each random variable."""
    _random_variable_args = (
        (random_variables.Bernoulli, {"probs": 0.5}),
        (random_variables.Beta, {"concentration0": 1, "concentration1": 1}),
        (random_variables.Binomial, {"total_count": 5.0, "probs": 0.5, "sample": 1}),
        (random_variables.Categorical, {"probs": [0.1, 0.5, 0.4]}),
        (random_variables.Cauchy, {"loc": 0, "scale": 1}),
        (random_variables.Chi2, {"df": 2}),
        (random_variables.Constant, {"loc": 3}),
        (random_variables.Dirichlet, {"concentration": [1, 2], "sample": [0.5, 0.5]}),
        (random_variables.DiscreteUniform, {"low": 2, "high": 10, "sample": 5}),
        (random_variables.Exponential, {"rate": 1}),
        (random_variables.Gamma, {"concentration": 3.0, "rate": 2.0}),
        (random_variables.Geometric, {"probs": 0.5, "sample": 10}),
        (random_variables.Gumbel, {"loc": 0, "scale": 1}),
        (random_variables.HalfCauchy, {"loc": 0, "scale": 1}),
        (random_variables.HalfNormal, {"scale": 3.0}),
        (random_variables.HalfStudentT, {"loc": 0, "scale": 1, "df": 10}),
        (random_variables.InverseGamma, {"concentration": 3, "rate": 2}),
        (random_variables.InverseGaussian, {"loc": 1, "concentration": 1}),
        (random_variables.Kumaraswamy, {"concentration0": 0.5, "concentration1": 0.5}),
        (random_variables.LKJ, {"dimension": 1, "concentration": 1.5, "sample": [[1]]}),
        (random_variables.Laplace, {"loc": 0, "scale": 1}),
        (random_variables.LogNormal, {"loc": 0, "scale": 1}),
        (random_variables.Logistic, {"loc": 0, "scale": 3}),
        (
            random_variables.Multinomial,
            {"total_count": 4, "probs": [0.2, 0.3, 0.5], "sample": [1, 1, 2]},
        ),
        (random_variables.LogitNormal, {"loc": 0, "scale": 1}),
        (
            random_variables.MultivariateNormalFullCovariance,
            {"loc": [1, 2], "covariance_matrix": [[0.36, 0.12], [0.12, 0.36]], "sample": [1, 2]},
        ),
        (random_variables.NegativeBinomial, {"total_count": 5, "probs": 0.5, "sample": 5}),
        (random_variables.Normal, {"loc": 0, "scale": 1}),
        (random_variables.Pareto, {"concentration": 1, "scale": 0.1, "sample": 5}),
        (random_variables.Poisson, {"rate": 2}),
        (random_variables.StudentT, {"loc": 0, "scale": 1, "df": 10}),
        (random_variables.Triangular, {"low": 0.0, "high": 1.0, "peak": 0.5}),
        (random_variables.Uniform, {"low": 0, "high": 1}),
        (random_variables.VonMises, {"loc": 0, "concentration": 1}),
        (random_variables.Weibull, {"scale": 0.1, "concentration": 1.0}),
        (random_variables.Wishart, {"df": 3, "scale_tril": [[1]], "sample": [[1]]}),
        (
            random_variables.ZeroInflatedBinomial,
            {"mix": 0.2, "total_count": 10, "probs": 0.5, "sample": 0},
        ),
        (
            random_variables.ZeroInflatedNegativeBinomial,
            {"mix": 0.2, "total_count": 10, "probs": 0.5, "sample": 0},
        ),
        (random_variables.ZeroInflatedPoisson, {"mix": 0.2, "rate": 2, "sample": 0}),
    )

    ids = [dist[0].__name__ for dist in _random_variable_args]
    return {
        "argnames": ("randomvariable", "kwargs"),
        "argvalues": _random_variable_args,
        "ids": ids,
    }


def test_tf_session_cleared(tf_session):
    """Check that fixture is finalizing correctly"""
    ops = tf_session.graph.get_operations()
    assert len(ops) == 0


@pytest.mark.parametrize(**random_variable_args())
def test_rvs_logp_and_forward_sample(tf_session, randomvariable, kwargs):
    """Test forward sampling and evaluating the logp for all random variables."""
    sample = kwargs.pop("sample", 0.1)
    dist = randomvariable(name="test_dist", **kwargs, validate_args=True)

    if randomvariable.__name__ not in ["Binomial", "ZeroInflatedBinomial"]:
        # Assert that values are returned with no exceptions
        log_prob = dist.log_prob()
        vals = tf_session.run([log_prob], feed_dict={dist._backend_tensor: sample})
        assert vals is not None

    else:
        # TFP issue ticket for Binom.sample_n https://github.com/tensorflow/probability/issues/81
        assert randomvariable.__name__ in ["Binomial", "ZeroInflatedBinomial"]
        with pytest.raises(NotImplementedError) as err:
            dist.log_prob()
            assert "NotImplementedError: sample_n is not implemented: Binomial" == str(err)


def test_rvs_backend_arithmetic(tf_session):
    """Test backend arithmetic implemented by the `WithBackendArithmetic` class."""
    x = random_variables.Normal("x", loc=0, scale=1)
    y = random_variables.Normal("y", loc=1, scale=2)

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
