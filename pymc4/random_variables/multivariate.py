"""
PyMC4 multivariate random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys
from tensorflow_probability import distributions as tfd

from .random_variable import RandomVariable, ContinuousRV


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = ["Dirichlet", "LKJ", "Multinomial", "MultivariateNormalFullCovariance", "Wishart"]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (ContinuousRV,), {"_base_dist": getattr(tfd, dist_name)}),
    )
