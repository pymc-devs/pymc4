# PyMC4 (Pre-release)

[![Build Status](https://dev.azure.com/pymc-devs/pymc4/_apis/build/status/pymc-devs.pymc4?branchName=master)](https://dev.azure.com/pymc-devs/pymc4/_build/latest?definitionId=1&branchName=master)
[![Coverage Status](https://codecov.io/gh/pymc-devs/pymc4/branch/master/graph/badge.svg)](https://codecov.io/gh/pymc-devs/pymc4)

High-level interface to TensorFlow Probability. Do not use for anything serious.

What works?

 * Build most models you could build with PyMC3
 * Sample using NUTS, all in TF, fully vectorized across chains (multiple chains basically become free)
 * Automatic transforms of model to the real line
 * Prior and posterior predictive sampling
 * Deterministic variables
 * Trace that can be passed to ArviZ

However, expect things to break or change without warning.

See here for an example: https://github.com/pymc-devs/pymc4/blob/master/notebooks/radon_hierarchical.ipynb
See here for the design document: https://github.com/pymc-devs/pymc4/blob/master/notebooks/pymc4_design_guide.ipynb
