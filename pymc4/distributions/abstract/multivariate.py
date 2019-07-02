"""
PyMC4 multivariate random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# pylint: disable=undefined-all-variable
from pymc4.distributions.abstract.distribution import (
    SimplexContinuousDistribution,
    DiscreteDistribution,
    ContinuousDistribution,
)


class Dirichlet(SimplexContinuousDistribution):
    r"""
    Dirichlet random variable.

    The Dirichlet distribution is used to model the probability distribution
    of the probability parameters of a Multinomial distribution. We basically
    use it in the same way as the Beta distribution is used to model the
    probability distribution of the probability parameter of a Bernoulli or
    Binomial distribution.

    The Dirichlet distribution is parameterized by ``a``, which is an
    array-like concentration parameter. It should be an array-like object of
    length K, where K is the number of multinomial classes. If values are
    lower than 1 you push probability mass to the edges and if larger than 1
    you push probability mass to the center. If it is equal to 1, we get a
    uniform distribution. Hence, the concentration parameter controls how flat
    or wide credibility is assigned within each of the multinomial classes.

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

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - a: concentration
    """

    def __init__(self, name, a, **kwargs):
        super().__init__(name, a=a, **kwargs)


class LKJ(ContinuousDistribution):
    r"""
    The LKJ (Lewandowski, Kurowicka and Joe) random variable.

    The LKJ distribution is a prior distribution for correlation matrices.
    If eta = 1 this corresponds to the uniform distribution over correlation
    matrices. For eta -> oo the LKJ prior approaches the identity matrix.

    ========  ==============================================
    Support   Upper triangular matrix with values in [-1, 1]
    ========  ==============================================

    Parameters
    ----------
    n : int
        Dimension of the covariance matrix (n > 1).
    eta : float
        The shape parameter (eta > 0) of the LKJ distribution. eta = 1
        implies a uniform distribution of the correlation matrices;
        larger values put more weight on matrices with few correlations.

    References
    ----------
    .. [LKJ2009] Lewandowski, D., Kurowicka, D. and Joe, H. (2009).
        "Generating random correlation matrices based on vines and
        extended onion method." Journal of multivariate analysis,
        100(9), pp.1989-2001.

    Developer Notes
    ---------------
    Unlike PyMC3's implementation, the LKJ distribution in PyMC4 returns fully
    populated covariance matrices, rather than upper triangle matrices.

    Parameter mappings to TensorFlow Probability are as follows:

    - n: dimension
    - eta: concentration
    """

    def __init__(self, name, n, eta, **kwargs):
        super().__init__(name, n=n, eta=eta, **kwargs)


class Multinomial(DiscreteDistribution):
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

    def __init__(self, name, n, p, **kwargs):
        super().__init__(name, n=n, p=p, **kwargs)


class MvNormal(ContinuousDistribution):
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

    Developer Notes
    ---------------
    ``MvNormal`` is based on TensorFlow Probability's
    ``MutivariateNormalFullCovariance``, in which the full covariance matrix
    must be specified.

    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - cov: covariance_matrix
    """

    def __init__(self, name, mu, cov, **kwargs):
        super().__init__(name, mu=mu, cov=cov, **kwargs)


class VonMisesFisher(ContinuousDistribution):
    r"""
    Von Mises-Fisher random variable.

    Generalizes Von Mises distribution to more than one dimension.

    .. math::
       f(x \mid \mu, \kappa) = \frac{(2 \pi)^{-n/2} \kappa^{n/2-1}}{I_{n/2-1}(\kappa)} exp(\kappa * mu^T x)

    where :math:`I_v(z)` is the modified Bessel function of the first kind of order v

    ========  =========================
    Support   :math:`x \in [-\pi, \pi]`
    Mean      :math:`\mu`
    ========  =========================

    Parameters
    ----------
    mu : array
        Mean.
    kappa : float
        Concentration (\frac{1}{kappa} is analogous to \sigma^2).

    Note
    ----
    Currently only n in {2, 3, 4, 5} are supported. For n=5 some numerical instability can
    occur for low concentrations (<.01).

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: mean_direction
    - kappa: concentration
    """

    def __init__(self, name, mu, kappa, **kwargs):
        super().__init__(name, mu=mu, kappa=kappa, **kwargs)


class Wishart(ContinuousDistribution):
    r"""
    Wishart random variable.

    The Wishart distribution is the probability distribution of the
    maximum-likelihood estimator (MLE) of the precision matrix of a
    multivariate normal distribution.  If V=1, the distribution is
    identical to the chi-square distribution with nu degrees of
    freedom.

    .. math::

       f(X \mid nu, T) =
           \frac{{\mid T \mid}^{nu/2}{\mid X \mid}^{(nu-k-1)/2}}{2^{nu k/2}
           \Gamma_p(nu/2)} \exp\left\{ -\frac{1}{2} Tr(TX) \right\}

    where :math:`k` is the rank of :math:`X`.

    ========  =========================================
    Support   :math:`X(p x p)` positive definite matrix
    Mean      :math:`nu V`
    Variance  :math:`nu (v_{ij}^2 + v_{ii} v_{jj})`
    ========  =========================================

    Parameters
    ----------
    nu : int
        Degrees of freedom, > 0.
    V : array
        p x p positive definite matrix.

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - nu: df
    - V: scale
    """

    def __init__(self, name, nu, V, **kwargs):
        super().__init__(name, nu=nu, V=V, **kwargs)
