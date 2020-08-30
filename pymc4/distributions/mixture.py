"""PyMC4 Distribution of a random variable consisting of a mixture of other
distributions.

Wraps tfd.Mixture as pm.Mixture
"""

import collections
from typing import Union, Tuple, List

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
    p : tf.Tensor
        p >= 0 and p <= 1
        The mixture weights, in the form of probabilities,
        must sum to one on the last (i.e., right-most) axis.
    distributions : pm.Distribution|sequence of pm.Distribution
        Multi-dimensional PyMC4 distribution (e.g. `pm.Poisson(...)`)
        or iterable of one-dimensional PyMC4 distributions
        :math:`f_1, \ldots, f_n`

    Examples
    --------
    Let's define a simple two-component Gaussian mixture:

    >>> import tensorflow as tf
    >>> import pymc4 as pm
    >>> @pm.model
    ... def mixture(dat):
    ...     p = tf.constant([0.5, 0.5])
    ...     m = yield pm.Normal("means", loc=tf.constant([0.0, 0.0]), scale=1.0)
    ...     comps = pm.Normal("comps", m, scale=1.0)
    ...     obs = yield pm.Mixture("mix", p=p, distributions=comps, observed=dat)
    ...     return obs

    The above implementation only allows components of the same family of distribitions.
    In order to allow for different families, we need a more verbose implementation:

    >>> @pm.model
    ... def mixture(dat):
    ...     p = tf.constant([0.5, 0.5])
    ...     m = yield pm.Normal("means", loc=tf.constant([0.0, 0.0]), scale=1.0)
    ...     comp1 = pm.Normal("comp1", m[..., 0], scale=1.0)
    ...     comp2 = pm.StudentT("comp2", m[..., 1], scale=1.0, df=3)
    ...     obs = yield pm.Mixture("mix", p=p, distributions=[comp1, comp2], observed=dat)
    ...     return obs

    We can also, as usual with Tensorflow, use higher dimensional parameters:

    >>> @pm.model
    ... def mixture(dat):
    ...     p = tf.constant([[0.8, 0.2], [0.4, 0.6], [0.5, 0.5]])
    ...     m = yield pm.Normal("means", loc=tf.constant([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), scale=1.0)
    ...     comp1 = pm.Normal("d1", m[..., 0], scale=1.0)
    ...     comp2 = pm.StudentT("d2", m[..., 1], scale=1.0, df=3)
    ...     obs = yield pm.Mixture("mix", p=p, distributions=[comp1, comp2], observed=dat)
    ...     return obs

    Note that in the last implementation the mixing weights need to sum to one
    on the right-most axis (to ensure correct parameterization use `validate_args=True`)
    """

    def __init__(
        self,
        name: str,
        p: tf.Tensor,
        distributions: Union[Distribution, List[Distribution], Tuple[Distribution]],
        **kwargs,
    ):
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
            distr = [el._distribution for el in d]
            return tfd.Mixture(
                tfd.Categorical(probs=p, **kwargs), distr, **kwargs, use_static_graph=True,
            )
        # else if 'd' is a pymc distribution with batch_size > 1
        elif isinstance(d, Distribution):
            return tfd.MixtureSameFamily(
                tfd.Categorical(probs=p, **kwargs), d._distribution, **kwargs
            )
        else:
            raise TypeError(
                "'distribution' needs to be a pymc4.Distribution object or a sequence of distributions"
            )
