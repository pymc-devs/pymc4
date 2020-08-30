"""Optimizers for ELBO convergence.

These optimizers wrap tf.optimizers with defaults from PyMC3.
"""
import tensorflow as tf


def adadelta(
    learning_rate: float = 1.0, rho: float = 0.95, epsilon: float = 1e-6, **kwargs
) -> tf.optimizers.Adadelta:
    r"""Adadelta optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    rho : float
        Squared gradient moving average decay factor
    epsilon : float
        Small value added for numerical stability
    
    Returns
    -------
    tf.optimizers.Adadelta

    Notes
    -----
    rho should be between 0 and 1. A value of rho close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
    work for multiple datasets (MNIST, speech).

    In the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does
    not become 0).

    Using the step size eta and a decay factor rho the learning rate is
    calculated as:

    .. math::
       r_t &= \rho r_{t-1} + (1-\rho)*g^2\\
       \eta_t &= \eta \frac{\sqrt{s_{t-1} + \epsilon}}
                             {\sqrt{r_t + \epsilon}}\\
       s_t &= \rho s_{t-1} + (1-\rho)*(\eta_t*g)^2

    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """
    return tf.optimizers.Adadelta(learning_rate=learning_rate, rho=rho, epsilon=epsilon, **kwargs)


def adagrad(learning_rate: float = 1.0, epsilon: float = 1e-6, **kwargs) -> tf.optimizers.Adagrad:
    r"""Adagrad optimizer.

    Parameters
    ----------
    learning_rate : float or symbolic scalar
        Learning rate
    epsilon : float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    tf.optimizers.Adagrad

    Notes
    -----
    Using step size eta Adagrad calculates the learning rate for feature i at
    time step t as:

    .. math:: \eta_{t,i} = \frac{\eta}
       {\sqrt{\sum^t_{t^\prime} g^2_{t^\prime,i}+\epsilon}} g_{t,i}

    as such the learning rate is monotonically decreasing.

    Epsilon is not included in the typical formula, see [2]_.

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.

    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """
    return tf.optimizers.Adagrad(learning_rate=learning_rate, epsilon=epsilon, **kwargs)


def adam(
    learning_rate: float = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    **kwargs,
) -> tf.optimizers.Adam:
    """Adam optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    beta_1 : float
        Exponential decay rate for the first moment estimates
    beta_2 : float
        Exponential decay rate for the second moment estimates
    epsilon : float
        Constant for numerical stability

    Returns
    -------
    tf.optimizers.Adam

    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    return tf.optimizers.Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs,
    )


def adamax(
    learning_rate: float = 0.002,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    **kwargs,
) -> tf.optimizers.Adamax:
    """Adamax optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    beta_1 : float
        Exponential decay rate for the first moment estimates
    beta_2 : float
        Exponential decay rate for the second moment estimates
    epsilon : float
        Constant for numerical stability

    Returns
    -------
    tf.optimizers.Adamax

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    return tf.optimizers.Adamax(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs,
    )


def sgd(learning_rate: float = 1e-3, **kwargs) -> tf.optimizers.SGD:
    """SGD optimizer.

    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    learning_rate : float
        Learning rate

    Returns
    -------
    tf.optimizers.SGD
    """
    return tf.optimizers.SGD(learning_rate=learning_rate, **kwargs)
