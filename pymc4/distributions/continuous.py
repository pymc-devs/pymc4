"""PyMC4 continuous random variables for tensorflow."""
import math

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as bij
from tensorflow_probability.python.internal import distribution_util as dist_util
from pymc4.distributions.distribution import (
    ContinuousDistribution,
    PositiveContinuousDistribution,
    UnitContinuousDistribution,
    BoundedContinuousDistribution,
)
from .half_student_t import HalfStudentT as TFPHalfStudentT


__all__ = [
    "Beta",
    "Cauchy",
    "Chi2",
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
    "Flat",
    "HalfFlat",
    "VonMises",
    "HalfStudentT",
    "Weibull",
]


class Normal(ContinuousDistribution):
    r"""Univariate normal random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    The normal distribution can be parameterized either in terms of precision or
    standard deviation. The link between the two parametrizations is given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 1000)
        locs = [0., 0., 0., -2.]
        scales = [0.4, 1., 2., 0.4]
        for loc, scale in zip(locs, scales):
            pdf = st.norm.pdf(x, loc, scale)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(loc, scale))
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
    loc : float
        Location parameter. For the normal distribution, this is the mean.
    scale : float
        Scale parameter. For the normal distribution, this is the standard
        deviation (scale > 0).

    Examples
    --------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.Normal('x', loc=0, scale=10)
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.Normal(loc=loc, scale=scale)


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
        for scale in [0.4, 1., 2.]:
            pdf = st.halfnorm.pdf(x, scale=scale)
            plt.plot(x, pdf, label=r'$\sigma$ = {}'.format(scale))
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
    scale : float
        Scale parameter :math:`sigma` (``sigma`` > 0) (only required if ``tau`` is not specified).

    Examples
    --------
    .. code-block:: python

        @pm.model
        def model():
            x = pm.HalfNormal('x', scale=10)
    """

    def __init__(self, name, scale, **kwargs):
        super().__init__(name, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        scale = conditions["scale"]
        return tfd.HalfNormal(scale=scale)


class HalfStudentT(PositiveContinuousDistribution):
    r"""Half Student's T random variable.

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 5, 200)
        scales = [1., 1., 2., 1.]
        nus = [.5, 1., 1., 30.]
        for scale, df in zip(scales, nus):
            pdf = st.t.pdf(x, df=df, loc=0, scale=scale)
            plt.plot(x, pdf, label=r'$\sigma$ = {}, $\nu$ = {}'.format(scale, df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    ========  ========================

    Parameters
    ----------
    df : float
        Degrees of freedom, also known as normality parameter (df > 0).
    scale : float
        Scale parameter (scale > 0). Converges to the standard deviation as df
        increases. (only required if lam is not specified)

    Examples
    --------
    .. code-block:: python

        @pm.model
        def model():
            x = pm.HalfStudentT('x', scale=10, df=10)

    In PyMC3, HalfStudentT's location was always zero. However, in a future PR, this can be changed.
    """

    def __init__(self, name, df, scale, **kwargs):
        super().__init__(name, df=df, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        scale = conditions["scale"]
        df = conditions["df"]
        return TFPHalfStudentT(df=df, loc=0, scale=scale)


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
    concentration0 : float
        concentration0 > 0.
    concentration1 : float
        concentration1 > 0.

    Notes
    -----
    Beta distribution is a conjugate prior for the parameter :math:`p` of
    the binomial distribution.
    """

    def __init__(self, name, concentration0, concentration1, **kwargs):
        super().__init__(
            name, concentration0=concentration0, concentration1=concentration1, **kwargs
        )

    @staticmethod
    def _init_distribution(conditions):
        concentration0, concentration1 = conditions["concentration0"], conditions["concentration1"]
        return tfd.Beta(concentration0=concentration0, concentration1=concentration1)


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
    loc : float
        Location parameter
    scale : float
        Scale parameter > 0
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.Cauchy(loc=loc, scale=scale)


class Chi2(PositiveContinuousDistribution):
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
    df : int
        Degrees of freedom (df > 0).
    """

    def __init__(self, name, df, **kwargs):
        super().__init__(name, df=df, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        df = conditions["df"]
        return tfd.Chi2(df=df)


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
    rate : float
        Rate or inverse scale (rate > 0)
    """

    def __init__(self, name, rate, **kwargs):
        super().__init__(name, rate=rate, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        rate = conditions["rate"]
        return tfd.Exponential(rate=rate)


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
    concentration : float
        Shape parameter (concentration > 0).
    rate : float
        Rate parameter (rate > 0).
    """

    def __init__(self, name, concentration, rate, **kwargs):
        super().__init__(name, concentration=concentration, rate=rate, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        concentration, rate = conditions["concentration"], conditions["rate"]
        return tfd.Gamma(concentration=concentration, rate=rate)


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
        locs = [0., 4., -1.]
        betas = [2., 2., 4.]
        for loc, beta in zip(locs, betas):
            pdf = st.gumbel_r.pdf(x, loc=loc, scale=beta)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\beta$ = {}'.format(loc, beta))
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
    loc : float
        Location parameter.
    scale : float
        Scale parameter (scale > 0).
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.Gumbel(loc=loc, scale=scale)


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
    scale : float
        Scale parameter (scale > 0).

    Developer Notes
    ----------------
    In PyMC3, HalfCauchy's location was always zero. However, in a future PR, this can be changed.
    """

    def __init__(self, name, scale, **kwargs):
        super().__init__(name, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        scale = conditions["scale"]
        return tfd.HalfCauchy(loc=0, scale=scale)


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
    concentration : float
        Shape parameter (concentration > 0).
    scale : float
        Scale parameter (scale > 0).
    """

    def __init__(self, name, concentration, scale, **kwargs):
        super().__init__(name, concentration=concentration, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        concentration, scale = conditions["concentration"], conditions["scale"]
        return tfd.InverseGamma(concentration=concentration, scale=scale)


class InverseGaussian(PositiveContinuousDistribution):
    r"""InverseGaussian random variable.

    Parameters
    ----------
    loc : float
    concentration : float
    """

    def __init__(self, name, loc, concentration, **kwargs):
        super().__init__(name, loc=loc, concentration=concentration, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, concentration = conditions["loc"], conditions["concentration"]
        return tfd.InverseGaussian(loc=loc, concentration=concentration)


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
    concentration0 : float
        concentration0 > 0.
    concentration1 : float
        concentration1 > 0.
    """

    def __init__(self, name, concentration0, concentration1, **kwargs):
        super().__init__(
            name, concentration0=concentration0, concentration1=concentration1, **kwargs
        )

    @staticmethod
    def _init_distribution(conditions):
        concentration0, concentration1 = conditions["concentration0"], conditions["concentration1"]
        return tfd.Kumaraswamy(concentration0=concentration0, concentration1=concentration1)


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
        locs = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for loc, b in zip(locs, bs):
            pdf = st.laplace.pdf(x, loc=loc, scale=b)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $b$ = {}'.format(loc, b))
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
    loc : float
        Location parameter.
    scale : float
        Scale parameter (scale > 0).
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.Laplace(loc=loc, scale=scale)


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
        locs = [0., 0., 0., -2.]
        scales = [.4, 1., 2., .4]
        for loc, scale in zip(locs, scales):
            pdf = st.logistic.pdf(x, loc=loc, scale=scale)
            plt.plot(x, pdf, label=r'$\mu$ = {}, scale = {}'.format(loc, scale))
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
    loc : float
        Mean.
    scale : float
        Scale (scale > 0).
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.Logistic(loc=loc, scale=scale)


class LogitNormal(UnitContinuousDistribution):
    r"""LogitNormal random variable.

    Distribution of any random variable whose logit is normally distributed.
    If Y is normally distributed, and f(Y) is the standard logistic function,
    then X = f(Y) is logit-normally distributed.

    The logit-normal distribution can be used to model a proportion that is
    bound between 0 and 1, and where values of 0 and 1 never occur.

    Parameters
    ----------
    loc : float
        Location parameter.
    scale : float
        Standard deviation. (scale > 0).
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.LogitNormal(loc=loc, scale=scale)


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
        locs = [0., 0., 0.]
        scales = [.25, .5, 1.]
        for loc, scale in zip(locs, scales):
            pdf = st.lognorm.pdf(x, scale, scale=np.exp(loc))
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(loc, scale))
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
    loc : float
        Location parameter.
    scale : float
        Standard deviation. (scale > 0).

    Example
    -------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.Lognormal('x', loc=2, scale=30)
    """

    def __init__(self, name, loc, scale, **kwargs):
        super().__init__(name, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, scale = conditions["loc"], conditions["scale"]
        return tfd.LogNormal(loc=loc, scale=scale)


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
    concentration : float|tensor
        Shape parameter (concentration > 0).
    scale : float|tensor
        Scale parameter (scale > 0).
    """

    def __init__(self, name, concentration, scale, **kwargs):
        super().__init__(name, concentration=concentration, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        concentration, scale = conditions["concentration"], conditions["scale"]
        return tfd.Pareto(concentration=concentration, scale=scale)

    def upper_limit(self):
        return float("inf")

    def lower_limit(self):
        return self.conditions["scale"]

    @property
    def test_value(self):
        return (
            tf.zeros(self.batch_shape + self.event_shape, dtype=self.dtype)
            + self.conditions["scale"]
            + 1
        )


class StudentT(ContinuousDistribution):
    r"""Student's T random variable.

    Describes a normal variable whose precision is gamma distributed.
    If only df parameter is passed, this specifies a standard (central)
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
        locs = [0., 0., -2., -2.]
        scales = [1., 1., 1., 2.]
        dfs = [1., 5., 5., 5.]
        for loc, scale, df in zip(locs, scales, dfs):
            pdf = st.t.pdf(x, df, loc=loc, scale=scale)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\nu$ = {}'.format(loc, scale, df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    ========  ========================

    Parameters
    ----------
    df : float|tensor
        Degrees of freedom, also known as normality parameter (df > 0).
    loc : float|tensor
        Location parameter.
    scale : float|tensor
        Scale parameter (scale > 0). Converges to the standard deviation as df
        increases. (only required if lam is not specified)

    Examples
    --------
    .. code-block:: python

        @pm.model
        def model():
            x = pm.StudentT('x', df=15, loc=0, scale=10)
    """

    def __init__(self, name, loc, scale, df, **kwargs):
        super().__init__(name, loc=loc, scale=scale, df=df, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        df, loc, scale = conditions["df"], conditions["loc"], conditions["scale"]
        return tfd.StudentT(df=df, loc=loc, scale=scale)


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
    low : float|tensor
        Lower limit.
    peak: float|tensor
        mode
    high : float|tensor
        Upper limit.
    """

    def __init__(self, name, low, peak, high, **kwargs):
        super().__init__(name, low=low, peak=peak, high=high, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        low, high, peak = conditions["low"], conditions["high"], conditions["peak"]
        return tfd.Triangular(low=low, high=high, peak=peak)

    def lower_limit(self):
        return self.conditions["low"]

    def upper_limit(self):
        return self.conditions["high"]


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
    low : float|tensor
        Lower limit.
    high : float|tensor
        Upper limit.
    """

    def __init__(self, name, low, high, **kwargs):
        super().__init__(name, low=low, high=high, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        low, high = conditions["low"], conditions["high"]
        return tfd.Uniform(low=low, high=high)

    # FIXME should we rename this functions as well?
    def lower_limit(self):
        return self.conditions["low"]

    def upper_limit(self):
        return self.conditions["high"]


class Flat(ContinuousDistribution):
    r"""A uniform distribution with support :math:`(-\inf, \inf)`.
    Used as a uninformative log-likelihood that returns
    zeros regardless of the passed values.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        return tfd.Uniform(low=-np.inf, high=np.inf)

    def log_prob(self, value):
        # convert the value to tensor
        value = tf.convert_to_tensor(value)
        expected = tf.zeros(self.batch_shape + self.event_shape)
        # check if the event shape matches
        if len(self.event_shape) and value.shape[-len(self.event_shape) :] != self.event_shape:
            raise ValueError("values not consistent with the event shape of distribution")
        # broadcast expected to shape of value
        if len(value.shape) < len(self.batch_shape + self.event_shape):
            if (
                value.shape[: len(value.shape) - len(self.event_shape)]
                != self.batch_shape[len(self.batch_shape) - len(value.shape) :]
            ):
                raise ValueError(
                    "batch shape of values not consistent with distribution's batch shape"
                )
        else:
            try:
                expected = tf.broadcast_to(expected, value.shape)
            except tf.errors.InvalidArgumentError:
                raise ValueError(
                    "shape of value not consistent with the distribution's batch + event shape"
                )

        return tf.reduce_sum(expected, axis=range(-len(self._distribution.event_shape), 0))

    def sample(self, shape=(), seed=None):
        """Raises ValueError as it is not possible to sample
        from flat distribution.
        """
        raise TypeError("cannot sample from a flat distribution")


class HalfFlat(PositiveContinuousDistribution):
    r"""Improper flat priors over positive reals."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        return tfd.Uniform(low=0.0, high=np.inf)

    def log_prob(self, value):
        # convert the value to tensor
        value = tf.convert_to_tensor(value)
        value = tf.where(value > 0, x=0.0, y=-np.inf)
        expected = tf.zeros(self.batch_shape + self.event_shape)
        # check if the event shape matches
        if len(self.event_shape) and value.shape[-len(self.event_shape) :] != self.event_shape:
            raise ValueError("values not consistent with the event shape of distribution")
        # broadcast expected to shape of value
        if len(value.shape) < len(self.batch_shape + self.event_shape):
            expected = expected + value
            if (
                value.shape[: len(value.shape) - len(self.event_shape)]
                != self.batch_shape[len(self.batch_shape) - len(value.shape) :]
            ):
                raise ValueError(
                    "batch shape of values not consistent with distribution's batch shape"
                )
        else:
            try:
                expected = tf.broadcast_to(expected, value.shape) + value
            except tf.errors.InvalidArgumentError:
                raise ValueError(
                    "shape of value not consistent with the distribution's batch + event shape"
                )

        return tf.reduce_sum(expected, axis=range(-len(self.event_shape), 0))

    def sample(self, shape=(), seed=None):
        """Raises ValueError as it is not possible to sample
        from half flat distribution.
        """
        raise TypeError("cannot sample from a half flat distribution")


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
        locs = [0., 0., 0.,  -2.5]
        concentrations = [.01, 0.5,  4., 2.]
        for loc, concentration in zip(locs, concentrations):
            pdf = st.vonmises.pdf(x, concentration, loc=loc)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\kappa$ = {}'.format(loc, concentration))
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
    loc : float|tensor
        Mean.
    concentration : float|tensor
        Concentration (\frac{1}{kappa} is analogous to \sigma^2).
    """

    def __init__(self, name, loc, concentration, **kwargs):
        super().__init__(name, loc=loc, concentration=concentration, **kwargs)

    @staticmethod
    def _init_distribution(conditions):
        loc, concentration = conditions["loc"], conditions["concentration"]
        return tfd.VonMises(loc=loc, concentration=concentration)

    def lower_limit(self):
        return -math.pi

    def upper_limit(self):
        return math.pi


class Weibull(PositiveContinuousDistribution):
    r"""Weibull random variable.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 200)
        concentrations = [.5, 1., 1.5, 5., 5.]
        scales = [1., 1., 1., 1.,  2]
        for concentration, scale in zip(concentrations, scales):
            pdf = st.weibull_min.pdf(x, concentration, scale)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(concentration, scale))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 2.5)
        plt.legend(loc=1)
        plt.show()

    ========  ====================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
    Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2)`
    ========  ====================================================

    Parameters
    ----------
     concentration : float|tensor
        Shape parameter (concentration > 0).
    scale : float|tensor
        Scale parameter (scale > 0).
    
    Developer Notes
    ---------------
    The Weibull distribution is implemented as a standard uniform distribution transformed by the
    Inverse of the WeibullCDF bijector. The shape to broadcast the low and high parameters for the
    Uniform distribution are obtained using 
    tensorflow_probability.python.internal.distribution_util.prefer_static_broadcast_shape()
    """

    def __init__(self, name, concentration, scale, **kwargs):
        super().__init__(name, concentration=concentration, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions):

        concentration, scale = conditions["concentration"], conditions["scale"]

        scale_tensor, concentration_tensor = (
            tf.convert_to_tensor(scale),
            tf.convert_to_tensor(concentration),
        )
        broadcast_shape = dist_util.prefer_static_broadcast_shape(
            scale_tensor.shape, concentration_tensor.shape
        )

        return tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=tf.zeros(broadcast_shape), high=tf.ones(broadcast_shape)),
            bijector=bij.Invert(bij.WeibullCDF(scale=scale, concentration=concentration)),
            name="Weibull",
        )
