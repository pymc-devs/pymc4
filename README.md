# NOTICE: Official development of this project has ceased, and it is no longer intended to become the next major version of PyMC. Ongoing development will continue on the PyMC3 project (pymc3-devs/pymc3).

See [the announcement](https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b) for more details on the future of PyMC and Theano.

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

## Develop

One easy way of developing on PyMC4 is to take advantage of the development containers! 
Using pre-built development environments allows you to develop on PyMC4 without needing to set up locally.

To use the dev containers, you will need to have Docker and VSCode running locally on your machine, 
and will need the VSCode Remote extension (`ms-vscode-remote.vscode-remote-extensionpack`).

Once you have done that, to develop on PyMC4, on GitHub:

1. Make a fork of the repository
2. Create a new branch inside your fork
3. Copy the branch URL

Now, in VSCode:

1. In the command palette, search for "Remote-Containers: Open Repository in Container...".
2. Paste in the branch URL
3. If prompted, create it in a "Unique Volume".

Happy hacking away! 
Because the repo will be cloned into an ephemeral repo,
**don't forget to commit your changes and push them to your branch!**
Then follow the usual pull request workflow back into PyMC4.

We hope you enjoy the time saved on setting up your development environment!
