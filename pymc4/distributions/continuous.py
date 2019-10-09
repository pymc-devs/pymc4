"""PyMC4 continuous random variables for tensorflow."""
import math

import tensorflow_probability as tfp
from pymc4.distributions.distribution import (
    ContinuousDistribution,
    PositiveContinuousDistribution,
    UnitContinuousDistribution,
    BoundedContinuousDistribution,
)


tfd = tfp.distributions

__all__ = [
    "Beta",
    "Cauchy",
    "ChiSquared",
    "Exponential",
    "Gamma",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "InverseGaussian",
    "Kumaraswamy",
    "Laplace",
    "LogNormal",
    "Logistic",
    "LogitNormal",
    "Normal",
    "Pareto",
    "StudentT",
    "Triangular",
    "Uniform",
    "VonMises",
]


class Normal(ContinuousDistribution):
    r"""Univariate normal random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 1000)
        mus = [0., 0., 0., -2.]
        sigmas = [0.4, 1., 2., 0.4]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.norm.pdf(x, mu, sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{1}{\tau}` or :math:`\sigma^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float|tensor
        Mean.
    sigma : float|tensor
        Standard deviation (sigma > 0).

    Examples
    --------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.Normal('x', mu=0, sigma=10)

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    """

    def __init__(self, name, mu, sigma, **kwargs):
        super().__init__(name, mu=mu, sigma=sigma, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, sigma = conditions["mu"], conditions["sigma"]
        return tfd.Normal(loc=mu, scale=sigma)


class HalfNormal(PositiveContinuousDistribution):
    r"""Half-normal random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \tau) =
           \sqrt{\frac{2\tau}{\pi}}
           \exp\left(\frac{-x^2 \tau}{2}\right)
       f(x \mid \sigma) =\sigma
           \sqrt{\frac{2}{\pi}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. note::

       The parameters ``sigma``/``tau`` (:math:`\sigma`/:math:`\tau`) refer to
       the standard deviation/precision of the unfolded normal distribution, for
       the standard deviation of the half-normal distribution, see below. For
       the half-normal, they are just two parameterisation :math:`\sigma^2
       \equiv \frac{1}{\tau}` of a scale parameter

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 5, 200)
        for sigma in [0.4, 1., 2.]:
            pdf = st.halfnorm.pdf(x, scale=sigma)
            plt.plot(x, pdf, label=r'$\sigma$ = {}'.format(sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\sqrt{\dfrac{2}{\tau \pi}}` or :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\dfrac{1}{\tau}\left(1 - \dfrac{2}{\pi}\right)` or :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    Parameters
    ----------
    sigma : float
        Scale parameter :math:`sigma` (``sigma`` > 0) (only required if ``tau`` is not specified).

    Examples
    --------
    .. code-block:: python

        @pm.model
        def model():
            x = pm.HalfNormal('x', sigma=10)

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - sigma: scale
    """

    def __init__(self, name, sigma, **kwargs):
        super().__init__(name, sigma=sigma, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        sigma = conditions["sigma"]
        return tfd.HalfNormal(scale=sigma)


class Beta(UnitContinuousDistribution):
    r"""Beta random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 1, 200)
        alphas = [.5, 5., 1., 2., 2.]
        betas = [.5, 1., 3., 2., 5.]
        for a, b in zip(alphas, betas):
            pdf = st.beta.pdf(x, a, b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 4.5)
        plt.legend(loc=9)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    ========  ==============================================================

    Parameters
    ----------
    alpha : float
        alpha > 0.
    beta : float
        beta > 0.

    Notes
    -----
    Beta distribution is a conjugate prior for the parameter :math:`p` of
    the binomial distribution.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: concentration0
    - beta: concentration1
    """

    def __init__(self, name, alpha, beta, **kwargs):
        super().__init__(name, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        alpha, beta = conditions["alpha"], conditions["beta"]
        return tfd.Beta(concentration0=alpha, concentration1=beta)


class Cauchy(ContinuousDistribution):
    r"""Cauchy random variable.

    Also known as the Lorentz or the Breit-Wigner distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 500)
        alphas = [0., 0., 0., -2.]
        betas = [.5, 1., 2., 1.]
        for a, b in zip(alphas, betas):
            pdf = st.cauchy.pdf(x, loc=a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mode      :math:`\alpha`
    Mean      undefined
    Variance  undefined
    ========  ========================

    Parameters
    ----------
    alpha : float
        Location parameter
    beta : float
        Scale parameter > 0

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:
    - alpha: loc
    - beta: scale
    """

    def __init__(self, name, alpha, beta, **kwargs):
        super().__init__(name, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        alpha, beta = conditions["alpha"], conditions["beta"]
        return tfd.Cauchy(loc=alpha, scale=beta)


class ChiSquared(PositiveContinuousDistribution):
    r""":math:`\chi^2` random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 15, 200)
        for df in [1, 2, 3, 6, 9]:
            pdf = st.chi2.pdf(x, df)
            plt.plot(x, pdf, label=r'$\nu$ = {}'.format(df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 0.6)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\nu`
    Variance  :math:`2 \nu`
    ========  ===============================

    Parameters
    ----------
    nu : int
        Degrees of freedom (nu > 0).

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:

    - nu: df

    The ChiSquared distribution name is copied over from PyMC3 for continuity. We map it to the
    Chi2 distribution in TensorFlow Probability.
    """

    def __init__(self, name, nu, **kwargs):
        super().__init__(name, nu=nu, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        nu = conditions["nu"]
        return tfd.Chi2(df=nu)


class Exponential(PositiveContinuousDistribution):
    r"""Exponential random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 100)
        for lam in [0.5, 1., 2.]:
            pdf = st.expon.pdf(x, scale=1.0/lam)
            plt.plot(x, pdf, label=r'$\lambda$ = {}'.format(lam))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{1}{\lambda}`
    Variance  :math:`\dfrac{1}{\lambda^2}`
    ========  ============================

    Parameters
    ----------
    lam : float
        Rate or inverse scale (lam > 0)

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:

    - lam: rate
    """

    def __init__(self, name, lam, **kwargs):
        super().__init__(name, lam=lam, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        lam = conditions["lam"]
        return tfd.Exponential(rate=lam)


class Gamma(PositiveContinuousDistribution):
    r"""Gamma random variable.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has mean beta.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 20, 200)
        alphas = [1., 2., 3., 7.5]
        betas = [.5, .5, 1., 1.]
        for a, b in zip(alphas, betas):
            pdf = st.gamma.pdf(x, a, scale=1.0/b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta}`
    Variance  :math:`\dfrac{\alpha}{\beta^2}`
    ========  ===============================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: concentration
    - beta: rate

    """

    def __init__(self, name, alpha, beta, **kwargs):
        super().__init__(name, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        alpha, beta = conditions["alpha"], conditions["beta"]
        return tfd.Gamma(concentration=alpha, rate=beta)


class Gumbel(ContinuousDistribution):
    r"""Univariate Gumbel random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \beta) = \frac{1}{\beta}e^{-(z + e^{-z})}

    where

    .. math::

        z = \frac{x - \mu}{\beta}.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        betas = [2., 2., 4.]
        for mu, beta in zip(mus, betas):
            pdf = st.gumbel_r.pdf(x, loc=mu, scale=beta)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\beta$ = {}'.format(mu, beta))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \beta\gamma`, where \gamma is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^2}{6} \beta^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    beta : float
        Scale parameter (beta > 0).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - beta: scale
    """

    def __init__(self, name, mu, beta, **kwargs):
        super().__init__(name, mu=mu, beta=beta, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, beta = conditions["mu"], conditions["beta"]
        return tfd.Gumbel(loc=mu, scale=beta)


class HalfCauchy(PositiveContinuousDistribution):
    r"""Half-Cauchy random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 5, 200)
        for b in [0.5, 1.0, 2.0]:
            pdf = st.cauchy.pdf(x, scale=b)
            plt.plot(x, pdf, label=r'$\beta$ = {}'.format(b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in [0, \infty)`
    Mode      0
    Mean      undefined
    Variance  undefined
    ========  ========================

    Parameters
    ----------
    beta : float
        Scale parameter (beta > 0).

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:

    - beta: scale

    In PyMC3, HalfCauchy's location was always zero. However, in a future PR, this can be changed.
    """

    def __init__(self, name, beta, **kwargs):
        super().__init__(name, beta=beta, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        beta = conditions["beta"]
        return tfd.HalfCauchy(loc=0, scale=beta)


class InverseGamma(PositiveContinuousDistribution):
    r"""Inverse gamma random variable, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 500)
        alphas = [1., 2., 3., 3.]
        betas = [1., 1., 1., .5]
        for a, b in zip(alphas, betas):
            pdf = st.invgamma.pdf(x, a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ======================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}`
              for :math:`\alpha > 2`
    ========  ======================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: concentration
    - beta: rate
    """

    def __init__(self, name, alpha, beta, **kwargs):
        super().__init__(name, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        alpha, beta = conditions["alpha"], conditions["beta"]
        return tfd.InverseGamma(concentration=alpha, scale=beta)


class InverseGaussian(PositiveContinuousDistribution):
    r"""InverseGaussian random variable.

    Parameters
    ----------
    mu : float
    lam : float

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - lam: concentration
    """

    def __init__(self, name, mu, lam, **kwargs):
        super().__init__(name, mu=mu, lam=lam, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, lam = conditions["mu"], conditions["lam"]
        return tfd.InverseGaussian(loc=mu, concentration=lam)


class Kumaraswamy(UnitContinuousDistribution):
    r"""Kumaraswamy random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid a, b) =
           abx^{a-1}(1-x^a)^{b-1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 1, 200)
        a_s = [.5, 5., 1., 2., 2.]
        b_s = [.5, 1., 3., 2., 5.]
        for a, b in zip(a_s, b_s):
            pdf = a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)
            plt.plot(x, pdf, label=r'$a$ = {}, $b$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 3.)
        plt.legend(loc=9)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`b B(1 + \tfrac{1}{a}, b)`
    Variance  :math:`b B(1 + \tfrac{2}{a}, b) - (b B(1 + \tfrac{1}{a}, b))^2`
    ========  ==============================================================

    Parameters
    ----------
    a : float
        a > 0.
    b : float
        b > 0.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - a: concentration0
    - b: concentration1

    """

    def __init__(self, name, a, b, **kwargs):
        super().__init__(name, a=a, b=b, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        a, b = conditions["a"], conditions["b"]
        return tfd.Kumaraswamy(concentration0=a, concentration1=b)


class Laplace(ContinuousDistribution):
    r"""Laplace random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-10, 10, 1000)
        mus = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for mu, b in zip(mus, bs):
            pdf = st.laplace.pdf(x, loc=mu, scale=b)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $b$ = {}'.format(mu, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2 b^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - b: scale
    """

    def __init__(self, name, mu, b, **kwargs):
        super().__init__(name, mu=mu, b=b, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, b = conditions["mu"], conditions["b"]
        return tfd.Laplace(loc=mu, scale=b)


class Logistic(ContinuousDistribution):
    r"""Logistic random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, s) =
           \frac{\exp\left(-\frac{x - \mu}{s}\right)}{s \left(1 + \exp\left(-\frac{x - \mu}{s}\right)\right)^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 200)
        mus = [0., 0., 0., -2.]
        ss = [.4, 1., 2., .4]
        for mu, s in zip(mus, ss):
            pdf = st.logistic.pdf(x, loc=mu, scale=s)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $s$ = {}'.format(mu, s))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\frac{s^2 \pi^2}{3}`
    ========  ==========================================


    Parameters
    ----------
    mu : float
        Mean.
    s : float
        Scale (s > 0).
    """

    def __init__(self, name, mu, s, **kwargs):
        super().__init__(name, mu=mu, s=s, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, s = conditions["mu"], conditions["s"]
        return tfd.Logistic(loc=mu, scale=s)


class LogitNormal(UnitContinuousDistribution):
    r"""LogitNormal random variable.

    Distribution of any random variable whose logit is normally distributed.
    If Y is normally distributed, and f(Y) is the standard logistic function,
    then X = f(Y) is logit-normally distributed.

    The logit-normal distribution can be used to model a proportion that is
    bound between 0 and 1, and where values of 0 and 1 never occur.

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Standard deviation. (sigma > 0).

    Developer Notes
    ---------------
    The logit-normal is implemented as Normal distribution transformed by the
    Sigmoid bijector.

    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc of tfd.Normal
    - sigma: scale of tfd.Normal
    """

    def __init__(self, name, mu, sigma, **kwargs):
        super().__init__(name, mu=mu, sigma=sigma, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, sigma = conditions["mu"], conditions["sigma"]
        return tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=mu, scale=sigma),
            bijector=tfp.bijectors.Sigmoid(),
            name="LogitNormal",
        )


class LogNormal(PositiveContinuousDistribution):
    r"""Log-normal random variable.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \frac{1}{x} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 100)
        mus = [0., 0., 0.]
        sigmas = [.25, .5, 1.]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.lognorm.pdf(x, sigma, scale=np.exp(mu))
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\exp\{\mu + \frac{1}{2\tau}\}`
    Variance  :math:`(\exp\{\frac{1}{\tau}\} - 1) \times \exp\{2\mu + \frac{1}{\tau}\}`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Standard deviation. (sigma > 0).

    Example
    -------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.Lognormal('x', mu=2, sigma=30)

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    """

    def __init__(self, name, mu, sigma, **kwargs):
        super().__init__(name, mu=mu, sigma=sigma, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, sigma = conditions["mu"], conditions["sigma"]
        return tfd.LogNormal(loc=mu, scale=sigma)


class Pareto(BoundedContinuousDistribution):
    r"""Pareto random variable.

    Often used to characterize wealth distribution, or other examples of the
    80/20 rule.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 4, 1000)
        alphas = [1., 2., 5., 5.]
        ms = [1., 1., 1., 2.]
        for alpha, m in zip(alphas, ms):
            pdf = st.pareto.pdf(x, alpha, scale=m)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, m = {}'.format(alpha, m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================================================
    Support   :math:`x \in [m, \infty)`
    Mean      :math:`\dfrac{\alpha m}{\alpha - 1}` for :math:`\alpha \ge 1`
    Variance  :math:`\dfrac{m \alpha}{(\alpha - 1)^2 (\alpha - 2)}`
              for :math:`\alpha > 2`
    ========  =============================================================

    Parameters
    ----------
    alpha : float|tensor
        Shape parameter (alpha > 0).
    m : float|tensor
        Scale parameter (m > 0).

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:

    - alpha: concentration
    - m: scale
    """

    def __init__(self, name, alpha, m, **kwargs):
        super().__init__(name, alpha=alpha, m=m, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        alpha, m = conditions["alpha"], conditions["m"]
        return tfd.Pareto(concentration=alpha, scale=m)

    def upper_limit(self):
        return float("inf")

    def lower_limit(self):
        return self.conditions["m"]


# TODO: Implement this
# class HalfStudentT(PositiveContinuousDistribution):
#     r"""Half Student's T random variable.

#     The pdf of this distribution is

#     .. math::

#         f(x \mid \sigma,\nu) =
#             \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
#             {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
#             \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

#     .. plot::

#         import matplotlib.pyplot as plt
#         import numpy as np
#         import scipy.stats as st
#         plt.style.use('seaborn-darkgrid')
#         x = np.linspace(0, 5, 200)
#         sigmas = [1., 1., 2., 1.]
#         nus = [.5, 1., 1., 30.]
#         for sigma, nu in zip(sigmas, nus):
#             pdf = st.t.pdf(x, df=nu, loc=0, scale=sigma)
#             plt.plot(x, pdf, label=r'$\sigma$ = {}, $\nu$ = {}'.format(sigma, nu))
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('f(x)', fontsize=12)
#         plt.legend(loc=1)
#         plt.show()

#     ========  ========================
#     Support   :math:`x \in [0, \infty)`
#     ========  ========================

#     Parameters
#     ----------
#     nu : float
#         Degrees of freedom, also known as normality parameter (nu > 0).
#     sigma : float
#         Scale parameter (sigma > 0). Converges to the standard deviation as nu
#         increases. (only required if lam is not specified)

#     Examples
#     --------
#     .. code-block:: python

#         # Only pass in one of lam or sigma, but not both.
#         @pm.model
#         def model():
#             x = pm.HalfStudentT('x', sigma=10, nu=10)

#     Developer Notes
#     ---------------
#     Parameter mappings to TensorFlow Probability are as follows:

#     - nu: df
#     - sigma: scale

#     In PyMC3, HalfStudentT's location was always zero. However, in a future PR, this can be changed.
#     """

#     def __init__(self, name, nu, sigma, **kwargs):
#         super().__init__(name, nu=nu, sigma=sigma, **kwargs)

#     @staticmethod
#     def _init_distribution(conditions):
#         nu, sigma = conditions["nu"], conditions["sigma"]
#         return tfd.HalfStudentT(df=nu, scale=sigma)


class StudentT(ContinuousDistribution):
    r"""Student's T random variable.

    Describes a normal variable whose precision is gamma distributed.
    If only nu parameter is passed, this specifies a standard (central)
    Student's T.

    The pdf of this distribution is

    .. math::

       f(x|\mu,\lambda,\nu) =
           \frac{\Gamma(\frac{\nu + 1}{2})}{\Gamma(\frac{\nu}{2})}
           \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
           \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-8, 8, 200)
        mus = [0., 0., -2., -2.]
        sigmas = [1., 1., 1., 2.]
        dfs = [1., 5., 5., 5.]
        for mu, sigma, df in zip(mus, sigmas, dfs):
            pdf = st.t.pdf(x, df, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\nu$ = {}'.format(mu, sigma, df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    ========  ========================

    Parameters
    ----------
    nu : float|tensor
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu : float|tensor
        Location parameter.
    sigma : float|tensor
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases. (only required if lam is not specified)

    Examples
    --------
    .. code-block:: python

        @pm.model
        def model():
            x = pm.StudentT('x', nu=15, mu=0, sigma=10)

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    - nu: df
    """

    def __init__(self, name, mu, sigma, nu, **kwargs):
        super().__init__(name, mu=mu, sigma=sigma, nu=nu, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        nu, mu, sigma = conditions["nu"], conditions["mu"], conditions["sigma"]
        return tfd.StudentT(df=nu, loc=mu, scale=sigma)


class Triangular(BoundedContinuousDistribution):
    r"""Continuous Triangular random variable.

    The pdf of this distribution is

    .. math::

       \begin{cases}
         0 & \text{for } x < a, \\
         \frac{2(x-a)}{(b-a)(c-a)} & \text{for } a \le x < c, \\[4pt]
         \frac{2}{b-a}             & \text{for } x = c, \\[4pt]
         \frac{2(b-x)}{(b-a)(b-c)} & \text{for } c < x \le b, \\[4pt]
         0 & \text{for } b < x.
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-2, 10, 500)
        lowers = [0., -1, 2]
        cs = [2., 0., 6.5]
        uppers = [4., 1, 8]
        for lower, c, upper in zip(lowers, cs, uppers):
            scale = upper - lower
            c_ = (c - lower) / scale
            pdf = st.triang.pdf(x, loc=lower, c=c_, scale=scale)
            plt.plot(x, pdf, label='lower = {}, c = {}, upper = {}'.format(lower,
                                                                           c,
                                                                           upper))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ============================================================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper + c}{3}`
    Variance  :math:`\dfrac{upper^2 + lower^2 +c^2 - lower*upper - lower*c - upper*c}{18}`
    ========  ============================================================================

    Parameters
    ----------
    lower : float|tensor
        Lower limit.
    c: float|tensor
        mode
    upper : float|tensor
        Upper limit.

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:

    - lower: low
    - c: peak
    - upper: high
    """

    def __init__(self, name, lower, c, upper, **kwargs):
        super().__init__(name, lower=lower, c=c, upper=upper, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        lower, upper, c = conditions["lower"], conditions["upper"], conditions["c"]
        return tfd.Triangular(low=lower, high=upper, peak=c)

    def lower_limit(self):
        return self.conditions["lower"]

    def upper_limit(self):
        return self.conditions["upper"]


class Uniform(BoundedContinuousDistribution):
    r"""Continuous uniform random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-3, 3, 500)
        ls = [0., -2]
        us = [2., 1]
        for l, u in zip(ls, us):
            y = np.zeros(500)
            y[(x<u) & (x>l)] = 1.0/(u-l)
            plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(loc=1)
        plt.show()

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower)^2}{12}`
    ========  =====================================

    Parameters
    ----------
    lower : float|tensor
        Lower limit.
    upper : float|tensor
        Upper limit.

    Developer Notes
    ----------------
    Parameter mappings to TensorFlow Probability are as follows:

    - lower: low
    - upper: high
    """

    def __init__(self, name, lower, upper, **kwargs):
        super().__init__(name, lower=lower, upper=upper, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        lower, upper = conditions["lower"], conditions["upper"]
        return tfd.Uniform(low=lower, high=upper)

    def lower_limit(self):
        return self.conditions["lower"]

    def upper_limit(self):
        return self.conditions["upper"]


class VonMises(BoundedContinuousDistribution):
    r"""Univariate VonMises random variable.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-np.pi, np.pi, 200)
        mus = [0., 0., 0.,  -2.5]
        kappas = [.01, 0.5,  4., 2.]
        for mu, kappa in zip(mus, kappas):
            pdf = st.vonmises.pdf(x, kappa, loc=mu)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\kappa$ = {}'.format(mu, kappa))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [-\pi, \pi]`
    Mean      :math:`\mu`
    Variance  :math:`1-\frac{I_1(\kappa)}{I_0(\kappa)}`
    ========  ==========================================

    Parameters
    ----------
    mu : float|tensor
        Mean.
    kappa : float|tensor
        Concentration (\frac{1}{kappa} is analogous to \sigma^2).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - kappa: concentration
    """

    def __init__(self, name, mu, kappa, **kwargs):
        super().__init__(name, mu=mu, kappa=kappa, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        mu, kappa = conditions["mu"], conditions["kappa"]
        return tfd.VonMises(loc=mu, concentration=kappa)

    def lower_limit(self):
        return -math.pi

    def upper_limit(self):
        return math.pi


# TODO: Implement this
# class Weibull(PositiveContinuousDistribution):
#     r"""Weibull random variable.

#     The pdf of this distribution is

#     .. math::

#        f(x \mid \alpha, \beta) =
#            \frac{\alpha x^{\alpha - 1}
#            \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

#     .. plot::

#         import matplotlib.pyplot as plt
#         import numpy as np
#         import scipy.stats as st
#         plt.style.use('seaborn-darkgrid')
#         x = np.linspace(0, 3, 200)
#         alphas = [.5, 1., 1.5, 5., 5.]
#         betas = [1., 1., 1., 1.,  2]
#         for a, b in zip(alphas, betas):
#             pdf = st.weibull_min.pdf(x, a, scale=b)
#             plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('f(x)', fontsize=12)
#         plt.ylim(0, 2.5)
#         plt.legend(loc=1)
#         plt.show()

#     ========  ====================================================
#     Support   :math:`x \in [0, \infty)`
#     Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
#     Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2)`
#     ========  ====================================================

#     Parameters
#     ----------
#     alpha : float|tensor
#         Shape parameter (alpha > 0).
#     beta : float|tensor
#         Scale parameter (beta > 0).

#     Developer Notes
#     ---------------
#     Parameter mappings to TensorFlow Probability are as follows:

#     - alpha: concentration
#     - beta: scale
#     """

#     def __init__(self, name, alpha, beta, **kwargs):
#         super().__init__(name, alpha=alpha, beta=beta, **kwargs)

#     @staticmethod
#     def _init_distribution(conditions):
#         alpha, beta = conditions["alpha"], conditions["beta"]
#         return tfd.Weibull(concentration=alpha, scale=beta)
