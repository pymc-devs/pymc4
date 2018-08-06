import functools

from tensorflow_probability.python.edward2.interceptor import interceptable
from tensorflow_probability.python.edward2.random_variable import RandomVariable

import pymc4.distributions as pmd

__all__ = [
    'HalfCauchy',
]


def _make_random_variable(distribution_cls):
    """Factory function to make random variable given distribution class."""
    @interceptable
    @functools.wraps(distribution_cls, assigned=("__module__", "__name__"))
    def func(*args, **kwargs):
        """Create a random variable for ${cls}.
        See ${cls} for more details.
        Returns:
        RandomVariable.
        #### Original Docstring for Distribution
        ${doc}
        """
        sample_shape = kwargs.pop("sample_shape", ())
        value = kwargs.pop("value", None)
        return RandomVariable(
            distribution=distribution_cls(*args, **kwargs),
            sample_shape=sample_shape,
            value=value )
    return func


# HalfCauchy = _make_random_variable(pmd.continuous.HalfCauchy)
HalfCauchy = _make_random_variable(pmd.HalfCauchy)
