"""Optimizers for ELBO convergence.

These optimizers wrap tf.optimizers with defaults from PyMC3.
"""
import tensorflow as tf


def adadelta(
    learning_rate: float = 1.0, rho: float = 0.95, epsilon: float = 1e-6
) -> tf.optimizers.Adadelta:
    """Adam optimizer.

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
    """
    return tf.optimizers.Adadelta(learning_rate=learning_rate, rho=rho, epsilon=epsilon)


def adagrad(learning_rate: float = 1.0, epsilon: float = 1e-6) -> tf.optimizers.Adagrad:
    """Adagrad optimizer.

    Parameters
    ----------
    learning_rate : float or symbolic scalar
        Learning rate
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    
    Returns
    -------
    tf.optimizers.Adagrad
    """
    return tf.optimizers.Adagrad(learning_rate=learning_rate, epsilon=epsilon)


def adam(
    learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8
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
    """
    return tf.optimizers.Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
    )


def adamax(
    learning_rate: float = 0.002, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8
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
    """
    return tf.optimizers.Adamax(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
    )


def sgd(learning_rate: float = 1e-3) -> tf.optimizers.SGD:
    """SGD optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate

    Returns
    -------
    tf.optimizers.SGD
    """
    return tf.optimizers.SGD(learning_rate=learning_rate)
