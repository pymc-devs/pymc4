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



class Bernoulli(RandomVariable):
    R"""Bernoulli random variable.

    The Bernoulli distribution describes the probability of successes
    (x=1) and failures (x=0).

    The pmf of this distribution is

    .. math:: f(x \mid p) = p^{x} (1-p)^{1-x}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = [0, 1]
        for p in [0, 0.5, 0.8]:
            pmf = st.bernoulli.pmf(x, p)
            plt.plot(x, pmf, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=9)
        plt.show()

    ========  ======================
    Support   :math:`x \in \{0, 1\}`
    Mean      :math:`p`
    Variance  :math:`p (1 - p)`
    ========  ======================

    Parameters
    ----------
    p : float
        Probability of success (0 < p < 1).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - p: probs
    """
    def _base_dist(self, p, *args, **kwargs):
        return tfd.Bernoulli(probs=p, *args, **kwargs)


class Binomial(RandomVariable):
    R"""
    Binomial random variable.

    The discrete probability distribution of the number of successes
    in a sequence of n independent yes/no experiments, each of which
    yields success with probability p.

    The pmf of this distribution is

    .. math:: f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 22)
        ns = [10, 17]
        ps = [0.5, 0.7]
        for n, p in zip(ns, ps):
            pmf = st.binom.pmf(x, n, p)
            plt.plot(x, pmf, '-o', label='n = {}, p = {}'.format(n, p))
        plt.xlabel('x', fontsize=14)
        plt.ylabel('f(x)', fontsize=14)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n p`
    Variance  :math:`n p (1 - p)`
    ========  ==========================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - n: total_count
    - p: probs
    """
    def _base_dist(self, n, p, *args, **kwargs):
        return tfd.Binomial(total_count=n, probs=p, *args, **kwargs)


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
tfp_supported = ["Geometric", "NegativeBinomial", "Poisson"]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
