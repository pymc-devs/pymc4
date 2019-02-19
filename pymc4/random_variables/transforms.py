"""
Convenience variable names for tfp Bijectors.
"""
from tensorflow_probability import bijectors


logodds = bijectors.Invert(bijectors.Sigmoid())
exp = bijectors.Exp
log = bijectors.Invert(bijectors.Exp())
identity = bijectors.Identity
