"""
PyMC4 multivariate random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys
from tensorflow_probability import distributions as tfd

from .random_variable import RandomVariable


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = ["Dirichlet", "LKJ", "Multinomial", "MultivariateNormalFullCovariance", "Wishart"]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )


# custom distributions
from ._multivariate.skew_normal import StdSkewNormal
from tensorflow_probability import bijectors as tfb


class SkewNormal(RandomVariable):
    """
    Skew Normal Distribution
    """
    def _base_dist(self, *args, **kwargs):
        name = kwargs['name']
        skew_kwargs = kwargs.setdefault('skew_kwargs', {})
        affine_kwargs = kwargs.setdefault('affine_kwargs', {})
        return tfd.TransformedDistribution(
            distribution=StdSkewNormal(corr=args[0], skew=args[1], **kwargs['skew_kwargs']),
            bijector=tfb.Affine(**affine_kwargs),
            name="SkewNormal",
        )