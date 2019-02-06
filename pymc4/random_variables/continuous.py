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


    This class is re-implemented so as to provide a familiar API to PyMC3 users.
    Here, we intentionally break TFD's style guide to make it easier to port
    models from PyMC3 to PyMC4.
    """

    def _base_dist(self, *args, **kwargs):
        try:
            loc = kwargs.pop("mu")
        except KeyError:
            loc = kwargs.pop("loc")

        try:
            scale = kwargs.pop("sigma")
        except KeyError:
            scale = kwargs.pop("scale")

        return tfd.Normal(loc=loc, scale=scale, **kwargs)


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
    "HalfNormal",
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
