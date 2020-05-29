"""PyMC4 discrete random variables."""
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from pymc4.distributions.distribution import (
    PositiveDiscreteDistribution,
    BoundedDiscreteDistribution,
)

__all__ = [
    "Bernoulli",
    "Binomial",
    "BetaBinomial",
    "DiscreteUniform",
    "Categorical",
    "Geometric",
    "NegativeBinomial",
    "OrderedLogistic",
    "Poisson",
    "Zipf",
]


class Bernoulli(BoundedDiscreteDistribution):
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
        for prob in [0, 0.5, 0.8]:
            pmf = st.bernoulli.pmf(x, prob)
            plt.plot(x, pmf, '-o', label='p = {}'.format(prob))
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
    probs : float
        Probability of success (0 < probs < 1).
    """

    def __init__(self, name, probs, **kwargs):
        super().__init__(name, probs=probs, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        probs = conditions["probs"]
        return tfd.Bernoulli(probs=probs)

    def lower_limit(self):
        return 0

    def upper_limit(self):
        return 1


class Binomial(BoundedDiscreteDistribution):
    r"""Binomial random variable.

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
        total_counts = [10, 17]
        probs = [0.5, 0.7]
        for total_count, prob in zip(total_counts, probs):
            pmf = st.binom.pmf(x, total_count, prob)
            plt.plot(x, pmf, '-o', label='n = {}, p = {}'.format(total_count, prob))
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
    total_count : int
        Number of Bernoulli trials (total_count >= 0).
    probs : float
        Probability of success in each trial (0 < probs < 1).
    """

    def __init__(self, name, total_count, probs, **kwargs):
        super().__init__(name, total_count=total_count, probs=probs, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        total_count, probs = conditions["total_count"], conditions["probs"]
        return tfd.Binomial(total_count=total_count, probs=probs)

    def lower_limit(self):
        return 0

    def upper_limit(self):
        return self.conditions["total_count"]


class BetaBinomial(BoundedDiscreteDistribution):
    r"""Bounded Discrete compound Beta-Binomial Random Variable

    The pmf of this distribution is

    .. math:: f(x \mid n, \alpha, \beta) = n \choose x \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}

    where :math:`n` = ``total_count``
          :math:`\alpha` = ``concentration0``
          :math:`\beta` = ``concentration1``

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        n = 10
        x = np.arange(0, n)
        alphas = [1, 2]
        betas = [6, 2]
        for a, b in zip(alphas, betas):
            pmf = st.betabinom(n, a, b).pmf(x)
            plt.plot(x, pmf, '-o', label='low = {}, high = {}'.format(low, high))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 0.4)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================================
    Support   :math:`x \in {0, 1 + 1, \ldots, n}`
    Mean      :math:`\dfrac{n \alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{n \alpha \beta (\alpha + \beta + n)}{(\alpha + \beta)^2 (\alpha + \beta + 1)}`
    ========  ===============================================

    Parameters
    ----------
    total_count : int
        Number of trials `n` (total_count>=0)
    concentration0 : float
        :math:`\alpha` parameter of the Beta Distribution (concentration0 > 0)
    concentration1 : float
        :math:`\beta` parameter of the Beta Distribution (concentration1 > 0)
    """

    def __init__(self, name, total_count, concentration0, concentration1, **kwargs):
        super().__init__(
            name,
            total_count=total_count,
            concentration0=concentration0,
            concentration1=concentration1,
            **kwargs,
        )

    @staticmethod
    def _init_distribution(conditions):
        total_count, concentration0, concentration1 = (
            conditions["total_count"],
            conditions["concentration0"],
            conditions["concentration1"],
        )
        return tfd.BetaBinomial(
            total_count=total_count, concentration0=concentration0, concentration1=concentration1
        )

    def lower_limit(self):
        return 0

    def upper_limit(self):
        return self.conditions["total_count"]


class DiscreteUniform(BoundedDiscreteDistribution):
    r"""Discrete uniform random variable.

    The pmf of this distribution is

    .. math:: f(x \mid low, high) = \frac{1}{high-low}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        lows = [1, -2]
        highs = [6, 2]
        for low, high in zip(lows, highs):
            x = np.arange(low, high+1)
            pmf = [1 / (high - low)] * len(x)
            plt.plot(x, pmf, '-o', label='low = {}, high = {}'.format(low, high))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 0.4)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================================
    Support   :math:`x \in {low, low + 1, \ldots, high}`
    Mean      :math:`\dfrac{low + high}{2}`
    Variance  :math:`\dfrac{(high - low)^2}{12}`
    ========  ===============================================

    Parameters
    ----------
    low : int
        Lower limit.
    high : int
        Upper limit (high > low).
    """

    def __init__(self, name, low, high, **kwargs):
        super().__init__(name, low=low, high=high, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        low, high = conditions["low"], conditions["high"]
        outcomes = tf.range(low, high + 1)
        return tfd.FiniteDiscrete(outcomes, probs=outcomes / (high - low))

    def lower_limit(self):
        return self.conditions["low"]

    def upper_limit(self):
        return self.conditions["high"]


class Categorical(BoundedDiscreteDistribution):
    r"""Categorical random variable.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        probs = [[0.1, 0.6, 0.3], [0.3, 0.1, 0.1, 0.5]]
        for prob in probs:
            x = range(len(prob))
            plt.plot(x, prob, '-o', label='p = {}'.format(prob))
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
    probs : array of floats
        probs > 0 and the elements of probs must sum to 1.
    """

    def __init__(self, name, probs, **kwargs):
        super().__init__(name, probs=probs, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        probs = tf.convert_to_tensor(conditions["probs"])
        outcomes = tf.range(probs.shape[-1])
        return tfd.FiniteDiscrete(outcomes, probs=probs)

    def lower_limit(self):
        return 0

    def upper_limit(self):
        return len(self.conditions["probs"])


class Geometric(BoundedDiscreteDistribution):
    r"""Geometric random variable.

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
        for prob in [0.1, 0.25, 0.75]:
            pmf = st.geom.pmf(x, prob)
            plt.plot(x, pmf, '-o', label='p = {}'.format(prob))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in \mathbb{N}_{\ge 0}`
    Mean      :math:`\dfrac{1}{p}`
    Variance  :math:`\dfrac{1 - p}{p^2}`
    ========  =============================

    Parameters
    ----------
    probs : float
        Probability of success on an individual trial (0 < probs <= 1).
    """
    # Another example for a wrong type used on the tensorflow side
    _test_value = 2.0  # type: ignore

    def __init__(self, name, probs, **kwargs):
        super().__init__(name, probs=probs, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        probs = conditions["probs"]
        return tfd.Geometric(probs=probs)

    def lower_limit(self):
        return 1

    def upper_limit(self):
        return float("inf")


class NegativeBinomial(PositiveDiscreteDistribution):
    r"""Negative binomial random variable.

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
        total_counts = [0.9, 2, 4]
        probs = [1, 2, 8]
        for total_count, prob in zip(total_counts, probs):
            pmf = NegBinom(total_count, prob, x)
            plt.plot(x, pmf, '-o', label=r'$n$ = {}, $p$ = {}'.format(total_count, prob))
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
    total_count : int
        The number of negative Bernoulli trials (i.e. failures) to stop at.
    probs : float
        Probability of success on an individual trial (0 < probs <= 1).
    """
    # For some ridiculous reason, tfp needs negative binomial values to be floats...
    _test_value = 0.0  # type: ignore

    def __init__(self, name, total_count, probs, **kwargs):
        super().__init__(name, total_count=total_count, probs=probs, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        total_count, probs = conditions["total_count"], conditions["probs"]
        return tfd.NegativeBinomial(total_count=total_count, probs=probs)


class Poisson(PositiveDiscreteDistribution):
    r"""Poisson random variable.

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
        for rate in [0.5, 3, 8]:
            pmf = st.poisson.pmf(x, rate)
            plt.plot(x, pmf, '-o', label='$\mu$ = {}'.format(rate))
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
    rate : float
        Expected number of occurrences during the given interval
        (rate >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """
    # For some ridiculous reason, tfp needs poisson values to be floats...
    _test_value = 0.0  # type: ignore

    def __init__(self, name, rate, **kwargs):
        super().__init__(name, rate=rate, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        rate = conditions["rate"]
        return tfd.Poisson(rate=rate)


# TODO: Implement this
# class ZeroInflatedBinomial(PositiveDiscreteDistribution):
#     r"""Zero-inflated Binomial log-likelihood.

#     The pmf of this distribution is

#     .. math::

#         f(x \mid \psi, n, p) = \left\{ \begin{array}{l}
#             (1-\psi) + \psi (1-p)^{n}, \text{if } x = 0 \\
#             \psi {n \choose x} p^x (1-p)^{n-x}, \text{if } x=1,2,3,\ldots,n
#             \end{array} \right.

#     .. plot::

#         import matplotlib.pyplot as plt
#         import numpy as np
#         import scipy.stats as st
#         plt.style.use('seaborn-darkgrid')
#         x = np.arange(0, 25)
#         ns = [10, 20]
#         ps = [0.5, 0.7]
#         psis = [0.7, 0.4]
#         for n, p, psi in zip(ns, ps, psis):
#             pmf = st.binom.pmf(x, n, p)
#             pmf[0] = (1 - psi) + pmf[0]
#             pmf[1:] =  psi * pmf[1:]
#             pmf /= pmf.sum()
#             plt.plot(x, pmf, '-o', label='n = {}, p = {}, $\\psi$ = {}'.format(n, p, psi))
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('f(x)', fontsize=12)
#         plt.legend(loc=1)
#         plt.show()

#     ========  ==========================
#     Support   :math:`x \in \mathbb{N}_0`
#     Mean      :math:`(1 - \psi) n p`
#     Variance  :math:`(1-\psi) n p [1 - p(1 - \psi n)].`
#     ========  ==========================

#     Parameters
#     ----------
#     psi : float
#         Expected proportion of Binomial variates (0 < psi < 1)
#     n : int
#         Number of Bernoulli trials (n >= 0).
#     p : float
#         Probability of success in each trial (0 < p < 1).
#     """

#     def __init__(self, name, psi, n, p, **kwargs):
#         super().__init__(name, psi=psi, n=n, p=p, **kwargs)

#     @staticmethod
#     def _init_distribution(conditions):
#         fill this in


# TODO: Implement this
# class ZeroInflatedNegativeBinomial(PositiveDiscreteDistribution):
#     r"""Zero-Inflated Negative binomial random variable.

#     The Zero-inflated version of the Negative Binomial (NB).
#     The NB distribution describes a Poisson random variable
#     whose rate parameter is gamma distributed.

#     The pmf of this distribution is

#     .. math::
#        f(x \mid \psi, \mu, \alpha) = \left\{
#          \begin{array}{l}
#            (1-\psi) + \psi \left (
#              \frac{\alpha}{\alpha+\mu}
#            \right) ^\alpha, \text{if } x = 0 \\
#            \psi \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} \left (
#              \frac{\alpha}{\mu+\alpha}
#            \right)^\alpha \left(
#              \frac{\mu}{\mu+\alpha}
#            \right)^x, \text{if } x=1,2,3,\ldots
#          \end{array}
#        \right.

#     .. plot::

#         import matplotlib.pyplot as plt
#         import numpy as np
#         import scipy.stats as st
#         from scipy import special
#         plt.style.use('seaborn-darkgrid')
#         def ZeroInfNegBinom(a, m, psi, x):
#             pmf = special.binom(x + a - 1, x) * (a / (m + a))**a * (m / (m + a))**x
#             pmf[0] = (1 - psi) + pmf[0]
#             pmf[1:] =  psi * pmf[1:]
#             pmf /= pmf.sum()
#             return pmf
#         x = np.arange(0, 25)
#         alphas = [2, 4]
#         mus = [2, 8]
#         psis = [0.7, 0.7]
#         for a, m, psi in zip(alphas, mus, psis):
#             pmf = ZeroInfNegBinom(a, m, psi, x)
#             plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\mu$ = {}, $\psi$ = {}'.format(a, m, psi))
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('f(x)', fontsize=12)
#         plt.legend(loc=1)
#         plt.show()

#     ========  ==========================
#     Support   :math:`x \in \mathbb{N}_0`
#     Mean      :math:`\psi\mu`
#     Var       :math:`\psi\mu +  \left (1 + \frac{\mu}{\alpha} + \frac{1-\psi}{\mu} \right)`
#     ========  ==========================

#     Parameters
#     ----------
#     psi : float
#         Expected proportion of NegativeBinomial variates (0 < psi < 1)
#     mu : float
#         Poission distribution parameter (mu > 0). Also corresponds to the number of expected
#         successes before the number of desired failures (alpha) is reached.
#     alpha : float
#         Gamma distribution parameter (alpha > 0). Also corresponds to the number of failures
#         desired.
#     """

#     def __init__(self, name, psi, mu, alpha, **kwargs):
#         super().__init__(name, psi=psi, mu=mu, alpha=alpha, **kwargs)


# TODO: Implement this
# class ZeroInflatedPoisson(PositiveDiscreteDistribution):
#     r"""
#     Zero-inflated Poisson random variable.

#     Often used to model the number of events occurring in a fixed period
#     of time when the times at which events occur are independent.

#     The pmf of this distribution is

#     .. math::

#         f(x \mid \psi, \theta) = \left\{ \begin{array}{l}
#             (1-\psi) + \psi e^{-\theta}, \text{if } x = 0 \\
#             \psi \frac{e^{-\theta}\theta^x}{x!}, \text{if } x=1,2,3,\ldots
#             \end{array} \right.

#     .. plot::

#         import matplotlib.pyplot as plt
#         import numpy as np
#         import scipy.stats as st
#         plt.style.use('seaborn-darkgrid')
#         x = np.arange(0, 22)
#         psis = [0.7, 0.4]
#         thetas = [8, 4]
#         for psi, theta in zip(psis, thetas):
#             pmf = st.poisson.pmf(x, theta)
#             pmf[0] = (1 - psi) + pmf[0]
#             pmf[1:] =  psi * pmf[1:]
#             pmf /= pmf.sum()
#             plt.plot(x, pmf, '-o', label='$\\psi$ = {}, $\\theta$ = {}'.format(psi, theta))
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('f(x)', fontsize=12)
#         plt.legend(loc=1)
#         plt.show()

#     ========  ==========================
#     Support   :math:`x \in \mathbb{N}_0`
#     Mean      :math:`\psi\theta`
#     Variance  :math:`\theta + \frac{1-\psi}{\psi}\theta^2`
#     ========  ==========================

#     Parameters
#     ----------
#     psi : float
#         Expected proportion of Poisson variates (0 < psi < 1)
#     theta : float
#         Expected number of occurrences during the given interval
#         (theta >= 0).
#     """

#     def __init__(self, name, psi, theta, **kwargs):
#         super().__init__(name, psi=psi, theta=theta, **kwargs)

#     @staticmethod
#     def _init_distribution(conditions):
#         fill this in


class Zipf(PositiveDiscreteDistribution):
    r"""Zipf random variable.

    The pmf of this distribution is

    .. math:: f(x \mid \alpha) = \frac{(x^(-\alpha))}{zeta(\alpha)}

    where zeta is the Riemann zeta function.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(1, 8)
        for power in [1.1, 2., 5.]:
            pmf = st.zipf.pmf(x, power)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}'.format(power))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=9)
        plt.show()

    ========  ===============================
    Support   :math:`x \in \{1,2,\ldots ,N\}`
    ========  ===============================

    Parameters
    ----------
    power : float
        Exponent parameter (power > 1).
    """

    def __init__(self, name, power, **kwargs):
        super().__init__(name, power=power, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        power = conditions["power"]
        return tfd.Zipf(power=power)


class OrderedLogistic(BoundedDiscreteDistribution):
    r"""Ordinal logistic random variable.

    An ordered discrete random variable. The OrderedLogistic
    distribution is parameterized by a location
    and a set of cutpoints. The pmf of this distribution

    .. math:: f(x \mid c, loc) = P(X > x-1) - P(x > x)
                                       = s(x-1; c, loc) - s(x; c, loc)

    where c is the set of cutpoints and s is the survival function of the
    distribution, i.e., the logistic function

    .. math:: s(x; c, loc) = logistic(loc - concat([-inf, c, inf])[x+1])

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |c| + 1}`
    ========  ===================================

    Parameters
    ----------
    loc : float
        mean of the latent logistic distribution
    cutpoints: array of floats
        array of cutpoints must be ordered such that cutpoints[i] <= cutpoints[i + 1]
    """

    def __init__(self, name, loc, cutpoints, **kwargs):
        super().__init__(name, loc=loc, cutpoints=cutpoints, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        cutpoints = tf.convert_to_tensor(conditions["cutpoints"])
        loc = conditions["loc"]
        return tfd.OrderedLogistic(cutpoints=cutpoints, loc=loc)

    def lower_limit(self):
        return 0

    def upper_limit(self):
        return len(self._distribution.cutpoints)
