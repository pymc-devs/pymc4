"""
Tests for PyMC4 auto naming.
"""

from .. import model

import pymc4 as pm
from pymc4 import Normal


def create_rvs(**kwargs):
    return pm.Normal(0, 1, name="test")


@pm.model(auto_name=True)
def autoname_test():
    mu = pm.Normal(0, 1)
    mu2 = Normal(0, 1)
    mu3 = pm.random_variables.Normal(0, 1)
    pm.Normal(0, 1, name="custom")
    dummy = pm.Normal(0, 1, name="custom2")
    x = create_rvs()


def test_auto_name():
    model = autoname_test.configure()

    rv_names = [rv.name for rv in model._forward_context.vars]
    expected_rv_names = ["mu", "mu2", "mu3", "custom", "custom2", "test"]

    assert rv_names == expected_rv_names
