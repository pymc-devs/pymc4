from pymc4.distributions.distribution import Distribution


__all__ = ["Mixture", "NormalMixture"]


class Mixture(Distribution):
    r"""
    Mixture random variable.
    Often used to model subpopulation heterogeneity
    .. math:: f(x \mid w, \theta) = \sum_{i = 1}^n w_i f_i(x \mid \theta_i)
    ========  ============================================
    Support   :math:`\cap_{i = 1}^n \textrm{support}(f_i)`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    ========  ============================================
    Parameters
    ----------
    p : array of floats
        p >= 0 and p <= 1
        the mixture weights, in the form of probabilities
    distributions : multidimensional PyMC4 distribution (e.g. `pm.Poisson(...)`)
        or iterable of one-dimensional PyMC4 distributions the
        component distributions :math:`f_1, \ldots, f_n`
    """
    pass


class NormalMixture(Distribution):
    r"""
    Normal mixture log-likelihood
    .. math::
        f(x \mid w, \mu, \sigma^2) = \sum_{i = 1}^n w_i N(x \mid \mu_i, \sigma^2_i)
    ========  =======================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    Variance  :math:`\sum_{i = 1}^n w_i^2 \sigma^2_i`
    ========  =======================================
    Parameters
    ----------
    w : array of floats
        w >= 0 and w <= 1
        the mixture weights
    loc : array of floats
        the component means
    scale : array of floats
        the component standard deviations
    """
    pass
