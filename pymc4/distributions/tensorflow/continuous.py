"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""
import tensorflow_probability as tfp
tfd = tfp.distributions


from .distribution import ContinuousDistribution

__all__ = ['Normal']


class Normal(ContinuousDistribution):
    r"""
    Univariate normal random variable.

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
        Standard deviation (sigma > 0).

    Examples
    --------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.Normal('x', mu=0, sigma=10)

    Developer Notes
    ---------------
    Parameter mappings to TensorFlow Probability are as follows:

    - mu: loc
    - sigma: scale
    """

    def __init__(self, name, mu, sigma, **kwargs):
        self.mu = mu
        self.sigma = sigma
        super().__init__(name, **kwargs)
        self._init_backend()

    def _init_backend(self):
        self._backend_dist = tfd.Normal(
            loc=self.mu, scale=self.sigma, name=self.name)

    def sample(self, shape=(), seed=None):
        return self._backend_dist.sample(shape, seed).numpy()

    def log_prob(self, value):
        return self._backend_dist.log_prob(value).numpy()


