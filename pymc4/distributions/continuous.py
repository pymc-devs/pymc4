import tensorflow_probability as tfp

__all__ = [
    "HalfCauchy"
]


class HalfCauchy(tfp.distributions.TransformedDistribution):  # pylint: disable=abstract-method
    def __init__(self, loc, scale, name, validate_args=False):
        super(HalfCauchy, self).__init__(
            distribution=tfp.distributions.Cauchy(loc, scale, validate_args=validate_args),
            bijector=tfp.bijectors.AbsoluteValue(validate_args=validate_args))

