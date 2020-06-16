"""PyMC4 Distribution of a random variable consisting of a mixture of other
distributions.

Wraps tfd.Mixture as pm.Mixture
"""

import tensorflow as tf
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
    p : array of floats
        p >= 0 and p <= 1
        the mixture weights, in the form of probabilities, must sum to one.
    distributions : multidimensional PyMC4 distribution (e.g. `pm.Poisson(...)`)
        or iterable of one-dimensional PyMC4 distributions the
        component distributions :math:`f_1, \ldots, f_n`
    """

    def __init__(self, name, p, distributions, **kwargs):
        super().__init__(name, p=p, distributions=distributions, **kwargs)

    @staticmethod
    def _init_distribution(conditions, **kwargs):
        p, d = conditions["p"], conditions["distributions"]
        # if 'd' is a list of pymc distributions, then use the underlying
        # tfp distributions for the mixture
        if isinstance(d, list):
            distributions = [el._distribution for el in d]
        # if 'd' is a pymc distribution with batch_size > 1, then build
        # K tfp distributions for tfd.mixture
        else:
            # get class of tfd distribution, i.e. Normal/Poisson/etc.
            ty = type(d._distribution)
            # construct list of tfp distributions
            distributions = []
            for i in range(d.batch_shape[0]):
                params = {}
                # build parameters for every component of the mixture
                for k, v in d.conditions.items():
                    v = tf.convert_to_tensor(v, dtype_hint="float32")
                    params[k] = v[i] if len(v.shape) else v
                # create new tfd distribution with parameter
                distributions.append(ty(**params))

        return tfd.Mixture(cat=tfd.Categorical(probs=p), components=distributions, **kwargs)
