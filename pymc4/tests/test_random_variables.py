"""
Tests for PyMC4 random variables
"""

from pymc4 import Normal
import numpy as np


def test_normal_dist(tf_session):
    """Small test of RandomVariable functionality.

    Test is intended to be deprecated and removed as full testing functionality is built out
    """

    normal_dist = Normal("test_normal", loc=0, scale=1)
    log_prob = normal_dist.log_prob()

    vals = tf_session.run([log_prob], feed_dict={normal_dist._backend_tensor: 0})
    assert np.isclose(vals[0], -0.918_938_5)


def test_tf_session_cleared(tf_session):
    """Temporary test: Check that fixture is finalizing correctly"""
    assert len(tf_session.graph.get_operations()) == 0
