import tensorflow as tf
from tensorflow_probability import edward2 as ed


__all__ = [
    'Input'
]


class InputDistribution(tf.contrib.distributions.Deterministic):  # pylint: disable=too-few-public-methods
    """
    detectable class for input
    is input <==> isinstance(rv.distribution, InputDistribution)
    """


def Input(name, shape, dtype=None):
    return ed.RandomVariable(InputDistribution(name=name, shape=shape, dtype=dtype))
