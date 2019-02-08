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
    r"""Bernoulli random variable.

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
    r"""
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
    def _base_dist(self, value, *args, **kwargs):
        return tfd.Deterministic(loc=value, *args, **kwargs)


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
    r"""
    Categorical random variable.

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


class Geometric(RandomVariable):
    r"""
    Geometric random variable.

    The probability that the first success in a sequence of Bernoulli
    trials occurs on the x'th trial.

    The pmf of this distribution is

    .. math:: f(x \mid p) = p(1-p)^{x-1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(1, 11)
        for p in [0.1, 0.25, 0.75]:
            pmf = st.geom.pmf(x, p)
            plt.plot(x, pmf, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in \mathbb{N}_{>0}`
    Mean      :math:`\dfrac{1}{p}`
    Variance  :math:`\dfrac{1 - p}{p^2}`
    ========  =============================

    Parameters
    ----------
    p : float
        Probability of success on an individual trial (0 < p <= 1).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - p: probs
    """

    def _base_dist(self, p, *args, **kwargs):
        return tfd.Geometric(probs=p, *args, **kwargs)


class NegativeBinomial(RandomVariable):
    r"""
    Negative binomial random variable.

    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.

    It is commonly used to model the number of Bernoulli trials needed until a
    fixed number of failures is reached.

    The pmf of this distribution is

    .. math::

       f(x \mid \mu, \alpha) =
           \binom{x + \alpha - 1}{x}
           (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        plt.style.use('seaborn-darkgrid')

        def NegBinom(a, m, x):
            pmf = special.binom(x + a - 1, x) * (a / (m + a))**a * (m / (m + a))**x
            return pmf

        x = np.arange(0, 22)
        alphas = [0.9, 2, 4]
        mus = [1, 2, 8]
        for a, m in zip(alphas, mus):
            pmf = NegBinom(a, m, x)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\mu$ = {}'.format(a, m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu : float
        Poission distribution parameter (mu > 0). Also corresponds to the number of expected
        successes before the number of desired failures (alpha) is reached.
    alpha : float
        Gamma distribution parameter (alpha > 0). Also corresponds to the number of failures
        desired.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu + alpha: total_count
    - mu / (mu + alpha): probs
    """

    def _base_dist(self, mu, alpha, *args, **kwargs):
        total_count = mu + alpha
        probs = mu / (mu + alpha)
        return tfd.NegativeBinomial(total_count=total_count, probs=probs, *args, **kwargs)


class Poisson(RandomVariable):
    r"""
    Poisson random variable.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 15)
        for m in [0.5, 3, 8]:
            pmf = st.poisson.pmf(x, m)
            plt.plot(x, pmf, '-o', label='$\mu$ = {}'.format(m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu : float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: rate
    """

    def _base_dist(self, mu, *args, **kwargs):
        return tfd.Poisson(rate=mu, *args, **kwargs)


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
    r"""
    Zero-inflated Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \theta) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\theta}, \text{if } x = 0 \\
            \psi \frac{e^{-\theta}\theta^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 22)
        psis = [0.7, 0.4]
        thetas = [8, 4]
        for psi, theta in zip(psis, thetas):
            pmf = st.poisson.pmf(x, theta)
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            plt.plot(x, pmf, '-o', label='$\\psi$ = {}, $\\theta$ = {}'.format(psi, theta))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\theta`
    Variance  :math:`\theta + \frac{1-\psi}{\psi}\theta^2`
    ========  ==========================

    Parameters
    ----------
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)
    theta : float
        Expected number of occurrences during the given interval
        (theta >= 0).
    """

    def _base_dist(self, psi, theta, *args, **kwargs):
        return tfd.Mixture(
            cat=Categorical(name="Categorical", p=[psi, 1.0 - psi])._distribution,
            components=[
                Constant(name="Constant", value=0)._distribution,
                Poisson(name="Poisson", mu=theta)._distribution,
            ],
            name="ZeroInflatedPoisson",
        )


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
# tfp_supported = [""]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
# for dist_name in tfp_supported:
#     setattr(
#         sys.modules[__name__],
#         dist_name,
#         type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
#     )
