"""
PyMC4 plots.

Plots are delegated to the ArviZ library, a general purpose library for
exploratory analysis of Bayesian models. See https://arviz-devs.github.io/arviz/
for details of plots.
"""

import arviz as az


autocorrplot = az.plot_autocorr
compareplot = az.plot_compare
forestplot = az.plot_forest
kdeplot = az.plot_kde
plot_posterior = az.plot_posterior
traceplot = az.plot_trace
energyplot = az.plot_energy
densityplot = az.plot_density
pairplot = az.plot_pair
