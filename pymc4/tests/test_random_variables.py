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
        "Bernoulli":{"probs": .5},
        "Beta": {"concentration0": .5, "concentration1": .5},
        "Binomial": {"total_count":5., "probs":.5},
        "Categorical":{"probs": [.1, .5, .4]},
        "Cauchy": {"loc":0, "scale":1},
        "Chi2": {"df":2},
        "Dirichlet": {"alpha": [1,2,3]},
        "Exponential":{"rate":1},
        "Gamma":{"concentration":3.0, "rate":2.0},
        "Geometric":{"probs":.5},
        "Gumbel":{"loc":0, "scale":1},
        "HalfCauchy":{"loc":0, "scale":1},
        "HalfNormal":{"loc":0, "scale":1},
        "InverseGamma":{"concentration":3, "rate":2},
        "InverseGaussian":{"loc":0, "concentration":1},
        "Kumaraswamy": {"concentration0": .5, "concentration1": .5},
        "LKJ":{"dimension":3, "concentration":1.5},
        "Laplace":{"loc":0, "scale":1},
        "LogNormal":{"loc":0, "scale":1},
        "Logistic":{"loc":0, "scale":3},
        "Multinomial":{"total_count":4, "probs":[.2,.3,.5]},
        "MultivariateNormalFullCovariance":{"loc":[1,2], "cov":[[.36, .12], [.12,.29]]},
        "NegativeBinomial":{"total_count":5, "probs":.5},
        "Normal":{"loc":0, "scale":1},
        "Pareto":{"concentration":0, "scale":1},
        "Poisson":{"rate":2},
        "StudentT":{"loc":0, "scale":1},
        "Triangular":{"low":3., "high":4., "peak":3.5},
        "Uniform":{"low":0, "high":1},
        "VonMises":{"loc":0, "concentration":1},
        "Wishart":{"df":5, "scale_tril":tf.cholesky(...)}
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


@pytest.mark.parametrize("tf_distribution",
                         _random_variables.tfp_supported,
                         ids=_random_variables.tfp_supported)
def test_tfp_distributions(tf_session, tf_supported_args, tf_distribution):
    """Test all TFP supported distributions"""

    _dist = getattr(_random_variables, tf_distribution)
    kwargs = tf_supported_args[tf_distribution]

    dist = _dist("test_dist", **kwargs, validate_args=True)
    log_prob = dist.log_prob()

    vals = tf_session.run([log_prob], feed_dict={dist._backend_tensor: 0})
    print(vals)
    assert vals != [-0.9189385]
