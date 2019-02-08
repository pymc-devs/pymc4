"""
PyMC4 discrete random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from .random_variable import RandomVariable
import numpy as np



class Constant(RandomVariable):
    _base_dist = tfd.Deterministic


# class DiscreteUniform(RandomVariable):
#     # def __init__(self, name, lower, upper, *args, **kwargs):
#     #     """Add `low` and `high` to kwargs."""
#     #     kwargs.update({"low": low, "high": high})
#     #     super(DiscreteUniform, self).__init__(name, *args, **kwargs)

#     def _base_dist(self, lower, upper, *args, **kwargs):
#         """
#         Discrete uniform base distribution.

#         A DiscreteUniform is an equiprobable Categorical over (upper - lower),
#         shifted up by low.
#         """
#         probs = np.ones(upper - lower).astype(int) / (upper - lower)
#         return tfd.TransformedDistribution(
#             distribution=tfd.Categorical(probs=probs),
#             bijector=tfp.bijectors.AffineScalar(shift=lower),
#             name="DiscreteUniform",
#         )

class Categorical(RandomVariable):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        ps = [[0.1, 0.6, 0.3], [0.3, 0.1, 0.1, 0.5]]
        for p in ps:
            x = range(len(p))
            plt.plot(x, p, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=1)
        plt.show()

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1. They will be automatically
        rescaled otherwise.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - p: probs
    """
    def _base_dist(self, p, *args, **kwargs):
        return tfd.Categorical(probs=p, *args, **kwargs)


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


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = ["Bernoulli", "Binomial", "Geometric", "NegativeBinomial", "Poisson"]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
