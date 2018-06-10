import tensorflow as tf
from tensorflow_probability import edward2 as ed


__all__ = [
    'Input'
]


class InputDistribution(tf.contrib.distributions.Deterministic):
    """
    detectable class for input
    is input <==> isinstance(rv.distribution, InputDistribution)
    """


def Input(name, shape, dtype=None):
    return ed.as_random_variable(InputDistribution(name=name, shape=shape, dtype=dtype))

