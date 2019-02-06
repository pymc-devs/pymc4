"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from .random_variable import RandomVariable


class HalfStudentT(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """
        Half student-T base distribution.

        A HalfStudentT is the absolute value of a StudentT.
        """
        return tfd.TransformedDistribution(
            distribution=tfd.StudentT(*args, **kwargs),
            bijector=tfp.bijectors.AbsoluteValue(),
            name="HalfStudentT",
        )


class LogitNormal(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """
        Logit normal base distribution.

        A LogitNormal is the standard logistic (i.e. sigmoid) of a Normal.
        """
        return tfd.TransformedDistribution(
            distribution=tfd.Normal(*args, **kwargs),
            bijector=tfp.bijectors.Sigmoid(),
            name="LogitNormal",
        )


class Normal(RandomVariable):
    """
    Univariate normal distribution.

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
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0) (only required if tau is not specified).

    Examples
    --------
    .. code-block:: python
        with pm.Model():
            x = pm.Normal('x', mu=0, sigma=10)
        with pm.Model():
            x = pm.Normal('x', mu=0, tau=1/23)

    Developer Notes
    ---------------

    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    """

    def _base_dist(self, mu, sigma, *args, **kwargs):
        return tfd.Normal(loc=mu, scale=sigma, **kwargs)


class HalfNormal(RandomVariable):
    """
    Half-normal log-likelihood.

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
        with pm.Model():
            x = pm.HalfNormal('x', sigma=10)

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - sigma: scale
    """

    def _base_dist(self, sigma, *args, **kwargs):
        return tfd.HalfNormal(scale=sigma, **kwargs)


class Weibull(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """
        Weibull base distribution.

        The inverse of the Weibull bijector applied to a U[0, 1] random
        variable gives a Weibull-distributed random variable.
        """
        return tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=0.0, high=1.0),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Weibull(*args, **kwargs)),
            name="Weibull",
        )


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = [
    "Beta",
    "Cauchy",
    "Chi2",
    "Exponential",
    "Gamma",
    "Gumbel",
    "HalfCauchy",
    # "HalfNormal",
    "InverseGamma",
    "InverseGaussian",
    "Kumaraswamy",
    "Laplace",
    "LogNormal",
    "Logistic",
    # "Normal",  # commented out to provide alternative parametrization.
    "Pareto",
    "StudentT",
    "Triangular",
    "Uniform",
    "VonMises",
]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
