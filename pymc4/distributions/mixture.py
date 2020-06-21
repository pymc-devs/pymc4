"""PyMC4 Distribution of a random variable consisting of a mixture of other
distributions.

Wraps tfd.Mixture as pm.Mixture
"""

import collections
from tensorflow_probability import distributions as tfd
from pymc4.distributions.distribution import Distribution


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
    p : array of floats|tensor
        p >= 0 and p <= 1
        The mixture weights, in the form of probabilities,
        must sum to one on the last (i.e. right-most) axis.
    distributions : PyMC4 distribution|sequence of PyMC4 distributions
        Multi-dimensional PyMC4 distribution (e.g. `pm.Poisson(...)`)
        or iterable of one-dimensional PyMC4 distributions
        :math:`f_1, \ldots, f_n`
    """

    def __init__(self, name, p, distributions, **kwargs):
        super().__init__(name, p=p, distributions=distributions, **kwargs)

    @staticmethod
    def _init_distribution(conditions, **kwargs):
        p, d = conditions["p"], conditions["distributions"]
        # if 'd' is a sequence of pymc distributions, then use the underlying
        # tfp distributions for the mixture
        if isinstance(d, collections.abc.Sequence):
            if any(not isinstance(el, Distribution) for el in d):
                raise TypeError(
                    "every element in 'distribution' needs to be a pymc4.Distribution object"
                )
            distr, mixture = [el._distribution for el in d], tfd.Mixture
        # else if 'd' is a pymc distribution with batch_size > 1
        elif isinstance(d, Distribution):
            distr, mixture = d._distribution, tfd.MixtureSameFamily
        else:
            raise TypeError(
                "'distribution' needs to be a pymc4.Distribution object or a sequence of distributions"
            )
        return mixture(tfd.Categorical(probs=p, **kwargs), distr, **kwargs)
