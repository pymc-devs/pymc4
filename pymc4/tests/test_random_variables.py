"""
Tests for PyMC4 random variables
"""
import pytest
import numpy as np
import tensorflow as tf


from .. import _random_variables


@pytest.fixture(scope="session")
def tf_supported_args():
    """Provide arugments for each supported Tensorflow distribution"""
    _tfp_supported_args = {
        "Bernoulli": {"probs": 0.5},
        "Beta": {"concentration0": 1, "concentration1": 1},
        "Binomial": {"total_count": 5.0, "probs": 0.5, "sample": 1},
        "Categorical": {"probs": [0.1, 0.5, 0.4]},
        "Cauchy": {"loc": 0, "scale": 1},
        "Chi2": {"df": 2},
        "Dirichlet": {"concentration": [1, 2], "sample": [0.5, 0.5]},
        "Exponential": {"rate": 1},
        "Gamma": {"concentration": 3.0, "rate": 2.0},
        "Geometric": {"probs": 0.5, "sample": 10},
        "Gumbel": {"loc": 0, "scale": 1},
        "HalfCauchy": {"loc": 0, "scale": 1},
        "HalfNormal": {"scale": 3.0},
        "InverseGamma": {"concentration": 3, "rate": 2},
        "InverseGaussian": {"loc": 1, "concentration": 1},
        "Kumaraswamy": {"concentration0": 0.5, "concentration1": 0.5},
        "LKJ": {"dimension": 1, "concentration": 1.5, "sample": [[1]]},
        "Laplace": {"loc": 0, "scale": 1},
        "LogNormal": {"loc": 0, "scale": 1},
        "Logistic": {"loc": 0, "scale": 3},
        "Multinomial": {"total_count": 4, "probs": [0.2, 0.3, 0.5], "sample": [1, 1, 2]},
        "MultivariateNormalFullCovariance": {
            "loc": [1, 2],
            "covariance_matrix": [[0.36, 0.12], [0.12, 0.36]],
            "sample": [1, 2],
        },
        "NegativeBinomial": {"total_count": 5, "probs": 0.5, "sample": 5},
        "Normal": {"loc": 0, "scale": 1},
        "Pareto": {"concentration": 1, "scale": 0.1, "sample": 5},
        "Poisson": {"rate": 2},
        "StudentT": {"loc": 0, "scale": 1, "df": 10},
        "Triangular": {"low": 0.0, "high": 1.0, "peak": 0.5},
        "Uniform": {"low": 0, "high": 1},
        "VonMises": {"loc": 0, "concentration": 1},
        "Wishart": {"df": 3, "scale_tril": [[1]], "sample": [[1]]},
    }

    return _tfp_supported_args


def test_normal_dist(tf_session):
    """Small test of RandomVariable functionality.

    Test is intended to be deprecated and removed as full testing functionality is built out
    """

    normal_dist = _random_variables.Normal("test_normal", loc=0, scale=1)
    log_prob = normal_dist.log_prob()

    vals = tf_session.run([log_prob], feed_dict={normal_dist._backend_tensor: 0})
    assert np.isclose(vals[0], -0.918_938_5)


def test_tf_session_cleared(tf_session):
    """Temporary test: Check that fixture is finalizing correctly"""
    assert len(tf_session.graph.get_operations()) == 0


@pytest.mark.parametrize(
    "tf_distribution", _random_variables.tfp_supported, ids=_random_variables.tfp_supported
)
def test_tfp_distributions(tf_session, tf_supported_args, tf_distribution):
    """Test all TFP supported distributions"""
    _dist = getattr(_random_variables, tf_distribution)

    kwargs = tf_supported_args[tf_distribution]
    sample = kwargs.pop("sample", 0.1)

    dist = _dist("test_dist", **kwargs, validate_args=True)

    if tf_distribution is not "Binomial":
        # Assert that values are returned with no exceptions
        log_prob = dist.log_prob()
        vals = tf_session.run([log_prob], feed_dict={dist._backend_tensor: sample})
        assert vals is not None

    else:

        # Bionomial distribution raises exception when calling log_prob
        assert tf_distribution == "Binomial"
        with pytest.raises(NotImplementedError) as err:
            dist.log_prob()
            assert "NotImplementedError: sample_n is not implemented: Binomial" == str(err)
