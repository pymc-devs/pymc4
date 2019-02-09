"""
PyMC4 multivariate random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys
from tensorflow_probability import distributions as tfd

from .random_variable import RandomVariable


class Dirichlet(RandomVariable):
    r"""
    Dirichlet random variable.

    The Dirichlet distribution is used to model the probability distribution
    of the probability parameters of a Multinomial distribution. We basically
    use it in the same way as the Beta distribution is used to model the
    probability distribution of the probability parameter of a Bernoulli or
    Binomial distribution.

    The Dirichlet distribution has only one parameter, a, which is a
    concentration parameter. It should be an array-like object of length
    K, where K is the number of multinomial classes. If values are lower than
    1 you push probability mass to the edges and if larger than 1 you push to
    the center. If it is equal to 1, we get a uniform distribution. Hence, the
    concentration parameter controls how flat or wide credibility is assigned
    within each of the multinomial classes.

    .. math::

       f(\mathbf{x}|\mathbf{a}) =
           \frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)}
           \prod_{i=1}^k x_i^{a_i - 1}

    ========  ===============================================
    Support   :math:`x_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}`
              such that :math:`\sum x_i = 1`
    Mean      :math:`\dfrac{a_i}{\sum a_i}`
    Variance  :math:`\dfrac{a_i - \sum a_0}{a_0^2 (a_0 + 1)}`
              where :math:`a_0 = \sum a_i`
    ========  ===============================================

    Parameters
    ----------
    a : array
        Concentration parameters (a > 0).
    """

    def _base_dist(self, a, *args, **kwargs):
        return tfd.Dirichlet(concentration=a, *args, **kwargs)


class Multinomial(RandomVariable):
    r"""
    Multinomial random variable.

    Generalizes binomial distribution, but instead of each trial resulting
    in "success" or "failure", each one results in exactly one of some
    fixed finite number k of possible outcomes over n independent trials.
    'x[i]' indicates the number of times outcome number i was observed
    over the n trials.

    .. math::

       f(x \mid n, p) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}

    ==========  ===========================================
    Support     :math:`x \in \{0, 1, \ldots, n\}` such that
                :math:`\sum x_i = n`
    Mean        :math:`n p_i`
    Variance    :math:`n p_i (1 - p_i)`
    Covariance  :math:`-n p_i p_j` for :math:`i \ne j`
    ==========  ===========================================

    Parameters
    ----------
    n : int or array
        Number of trials (n > 0). If n is an array its shape must be (N,) with
        N = p.shape[0]
    p : one- or two-dimensional array
        Probability of each one of the different outcomes. Elements must
        be non-negative and sum to 1 along the last axis.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - n: total_count
    - p: probs
    """

    def _base_dist(self, n, p, *args, **kwargs):
        return tfd.Multinomial(total_count=n, probs=p, *args, **kwargs)


class MvNormal(RandomVariable):
    r"""
    Multivariate normal random variable.

    .. math::
       f(x \mid \pi, T) =
           \frac{|T|^{1/2}}{(2\pi)^{k/2}}
           \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T (x-\mu) \right\}

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`T^{-1}`
    ========  ==========================

    Parameters
    ----------
    mu : array
        Vector of means.
    cov : array
        Covariance matrix.

    Examples
    --------
    Define a multivariate normal variable for a given covariance
    matrix::

        cov = np.array([[1., 0.5], [0.5, 2]])
        mu = np.zeros(2)
        vals = pm.MvNormal('vals', mu=mu, cov=cov, shape=(5, 2))

    Most of the time it is preferable to specify the cholesky
    factor of the covariance instead. For example, we could
    fit a multivariate outcome like this (see the docstring
    of `LKJCholeskyCov` for more information about this)::

        mu = np.zeros(3)
        true_cov = np.array([[1.0, 0.5, 0.1],
                             [0.5, 2.0, 0.2],
                             [0.1, 0.2, 1.0]])
        data = np.random.multivariate_normal(mu, true_cov, 10)
        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
        chol_packed = pm.LKJCholeskyCov('chol_packed',
            n=3, eta=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(3, chol_packed)
        vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=data)

    For unobserved values it can be better to use a non-centered
    parametrization::

        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
        chol_packed = pm.LKJCholeskyCov('chol_packed',
            n=3, eta=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(3, chol_packed)
        vals_raw = pm.Normal('vals_raw', mu=0, sigma=1, shape=(5, 3))
        vals = pm.Deterministic('vals', tt.dot(chol, vals_raw.T).T)

    Developer Notes
    ---------------
    MvNormal is based on TensorFlow Probability's
    MutivariateNormalFullCovariance, in which the full covariance matrix must
    be specified.

    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - cov: covariance_matrix
    """

    def _base_dist(self, mu, cov, *args, **kwargs):
        return tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov, *args, **kwargs)


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = ["LKJ", "Wishart"]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
