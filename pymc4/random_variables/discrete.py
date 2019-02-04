"""
PyMC4 discrete random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys
from tensorflow_probability import distributions as tfd

from .random_variable import RandomVariable


class Constant(RandomVariable):
    _base_dist = tfd.Deterministic


''' TODO DiscreteUniform raises a NotImplementedError from tfp.
class DiscreteUniform(RandomVariable):
    def __init__(self, name, low, high, *args, **kwargs):
        """Add `low` and `high` to kwargs."""
        kwargs.update({"low": low, "high": high})
        super(DiscreteUniform, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """
        Discrete uniform base distribution.

        A DiscreteUniform is an equiprobable Categorical over (high-low),
        shifted up by low.
        """
        low = kwargs.pop("low")
        high = kwargs.pop("high")
        probs = np.ones(high - low) / (high - low)
        return tfd.TransformedDistribution(
            distribution=tfd.Categorical(probs=probs),
            bijector=tfp.bijectors.AffineScalar(shift=low),
            name="DiscreteUniform",
        )
'''


class ZeroInflatedBinomial(RandomVariable):
    def __init__(self, name, mix, *args, **kwargs):
        """Add `mix` to kwargs."""
        kwargs.update({"mix": mix})
        super(ZeroInflatedBinomial, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """
        Zero-inflated binomial base distribution.

        A ZeroInflatedBinomial is a mixture between a deterministic
        distribution and a Binomial distribution.
        """
        mix = kwargs.pop("mix")
        return tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
            components=[tfd.Deterministic(0.0), tfd.Binomial(*args, **kwargs)],
            name="ZeroInflatedBinomial",
        )


class ZeroInflatedPoisson(RandomVariable):
    def __init__(self, name, mix, *args, **kwargs):
        """Add `mix` to kwargs."""
        kwargs.update({"mix": mix})
        super(ZeroInflatedPoisson, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """
        Zero-inflated Poisson base distribution.

        A ZeroInflatedPoisson is a mixture between a deterministic
        distribution and a Poisson distribution.
        """
        mix = kwargs.pop("mix")
        return tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
            components=[tfd.Deterministic(0.0), tfd.Poisson(*args, **kwargs)],
            name="ZeroInflatedPoisson",
        )


class ZeroInflatedNegativeBinomial(RandomVariable):
    def __init__(self, name, mix, *args, **kwargs):
        """Add `mix` to kwargs."""
        kwargs.update({"mix": mix})
        super(ZeroInflatedNegativeBinomial, self).__init__(name, *args, **kwargs)

    def _base_dist(self, *args, **kwargs):
        """
        Zero-inflated negative binomial base distribution.

        A ZeroInflatedNegativeBinomial is a mixture between a deterministic
        distribution and a NegativeBinomial distribution.
        """
        mix = kwargs.pop("mix")
        return tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, 1.0 - mix]),
            components=[tfd.Deterministic(0.0), tfd.NegativeBinomial(*args, **kwargs)],
            name="ZeroInflatedNegativeBinomial",
        )


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = ["Bernoulli", "Binomial", "Categorical", "Geometric", "NegativeBinomial", "Poisson"]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
