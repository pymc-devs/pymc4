"""
Statistical utility functions for PyMC4.

Diagnostics and auxiliary statistical functions are delegated to the ArviZ
library, a general purpose library for exploratory analysis of Bayesian models.
See https://arviz-devs.github.io/arviz/ for details.
"""

import sys
import arviz as az

# Access all ArviZ statistics
for stat in az.stats.__all__:
    setattr(sys.modules[__name__], stat, getattr(az.stats, stat))

__all__ = az.stats.__all__
