"""
Tests for PyMC4 random variables
"""
from collections import ChainMap
import pytest
import numpy as np

import pymc4 as pm
from .fixtures.fixtures_distributions import distribution, distribution_conditions, distribution_extra_parameters, broadcast_distribution, check_broadcast

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


def test_multivariate_normal_cholesky(tf_seed):
    mean = np.zeros(2)
    cov = np.array([[-1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match=r"Cholesky decomposition failed"):
        pm.MvNormal("x", loc=mean, covariance_matrix=cov)


def test_flat_halfflat_broadcast(tf_seed, check_broadcast):
    """Test the error messages returned by Flat and HalfFlat
    distributions for inconsistent sample shapes"""
    dist, samples = check_broadcast
    for sample in samples:
        with pytest.raises(ValueError, match=r"not consistent"):
            dist.log_prob(sample)


def test_extra_parameters(tf_seed, distribution_extra_parameters):
    distribution_name, arg_name, conditions, extra_parameters = distribution_extra_parameters
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
