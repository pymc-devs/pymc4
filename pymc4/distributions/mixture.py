from pymc4 import utils
from pymc4.distributions.distribution import Distribution
from tensorflow_probability import distributions as tfd
from collections.abc import Iterable


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

    def __init__(self, name, p, distributions, **kwargs):
        super().__init__(name, p=p, distributions=distributions, **kwargs)

    def _init_distribution(self, conditions, **kwargs):
        p, distributions = conditions["p"], conditions["distributions"]

        if not (
            (
                isinstance(distributions, Iterable)
                and all((isinstance(c, Distribution) for c in distributions))
            )
            or isinstance(distributions, Distribution)
        ):
            raise TypeError(
                "Supplied Mixture distributions must be a "
                "Distribution or an iterable of "
                "Distributions. Got {} instead.".format(
                    type(distributions)
                    if not isinstance(distributions, Iterable)
                    else [type(c) for c in distributions]
                )
            )

        if p.shape[:-1] != tuple(distributions[0]._distribution.batch_shape):
            raise ValueError(
                "`batch_shape` of categorical and component distributions should be equal"
            )

        if isinstance(distributions, Iterable):
            distributions = [_._distribution for _ in distributions]
            return tfd.Mixture(cat=tfd.Categorical(probs=p), components=distributions)
        else:
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=p),
                components_distribution=distributions._distribution,
            )


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

    def __init__(self, name, w, loc, scale, **kwargs):
        super().__init__(name, w=w, loc=loc, scale=scale, **kwargs)

    def _init_distribution(self, conditions, **kwargs):
        w, loc, scale = conditions["w"], conditions["loc"], conditions["scale"]

        if w.shape != loc.shape or w.shape != scale.shape:
            raise ValueError(
                "`batch_shape` of categorical and component distributions should be equal"
            )

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=w),
            components_distribution=tfd.Normal(loc=loc, scale=scale),
        )
