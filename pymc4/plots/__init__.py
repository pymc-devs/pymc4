"""
Plotting utility functions for PyMC4.

Plotting functions are delegated to the ArviZ library, a general purpose library
for exploratory analysis of Bayesian models. See
https://arviz-devs.github.io/arviz/ for details on plots.
"""
import sys
import arviz as az


# Access to arviz plots: base plots provided by arviz
for plot in az.plots.__all__:
    setattr(sys.modules[__name__], plot, (getattr(az.plots, plot)))


__all__ = az.plots.__all__
