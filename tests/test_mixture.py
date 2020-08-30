"""
Tests for PyMC4 mixture distribution
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import pymc4 as pm
from pymc4.coroutine_model import ModelTemplate

tfd = tfp.distributions

distribution_conditions = {
    "two_components": {
        "n": 1,
        "k": 2,
        "p": np.array([0.5, 0.5], dtype="float32"),
        "loc": np.array([0.0, 0.0], dtype="float32"),
        "scale": 1.0,
    },
    "three_components": {
        "n": 1,
        "k": 3,
        "p": np.array([0.5, 0.25, 0.25], dtype="float32"),
        "loc": np.array([0.0, 0.0, 0.0], dtype="float32"),
        "scale": 1.0,
    },
    "two_components_three_distributions": {
        "n": 3,
        "k": 2,
        "p": np.array([[0.5, 0.5], [0.8, 0.2], [0.7, 0.3]], dtype="float32"),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype="float32"),
        "scale": 1.0,
    },
    "three_components_three_distributions": {
        "n": 3,
        "k": 3,
        "p": np.array([[0.5, 0.25, 0.25], [0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], dtype="float32"),
        "loc": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype="float32"),
        "scale": 1.0,
    },
}


@pytest.fixture(scope="function", params=list(distribution_conditions), ids=str)
def mixture_components(request):
    par = distribution_conditions[request.param]
    return par["n"], par["k"], par["p"], par["loc"], par["scale"]


def _mixture(k, p, loc, scale, dat):
    m = yield pm.Normal("means", loc=loc, scale=scale)
    distributions = [pm.Normal("d" + str(i), loc=m[..., i], scale=scale) for i in range(k)]
    obs = yield pm.Mixture(
        "mixture", p=p, distributions=distributions, validate_args=True, observed=dat
    )
    return obs


def _mixture_same_family(k, p, loc, scale, dat):
    m = yield pm.Normal("means", loc=loc, scale=scale)
    distribution = pm.Normal("d", loc=m, scale=scale)
    obs = yield pm.Mixture(
        "mixture", p=p, distributions=distribution, validate_args=True, observed=dat
    )
    return obs


@pytest.fixture(scope="function", params=[_mixture_same_family, _mixture], ids=str)
def mixture(mixture_components, request):
    n, k, p, loc, scale = mixture_components
    dat = tfd.Normal(loc=np.zeros(n), scale=1).sample(100).numpy()
    if n == 1:
        dat = dat.reshape(-1)
    model = ModelTemplate(request.param, name="mixture", keep_auxiliary=True, keep_return=True)
    model = model(k, p, loc, scale, dat)
    return model, n, k


def test_wrong_distribution_argument_batched_fails():
    with pytest.raises(TypeError, match=r"sequence of distributions"):
        pm.Mixture("mix", p=[0.5, 0.5], distributions=tfd.Normal(0, 1))


def test_wrong_distribution_argument_in_list_fails():
    with pytest.raises(TypeError, match=r"every element in 'distribution' "):
        pm.Mixture(
            "mix",
            p=[0.5, 0.5],
            distributions=[pm.Normal("comp1", loc=0.0, scale=1.0), "not a distribution",],
        )


def test_sampling(mixture, xla_fixture):
    model, n, k = mixture
    if xla_fixture:
        with pytest.raises(tf.errors.InvalidArgumentError):
            pm.sample(model, num_samples=100, num_chains=2, xla=xla_fixture)
    else:
        trace = pm.sample(model, num_samples=100, num_chains=2, xla=xla_fixture)
        if n == 1:
            assert trace.posterior["mixture/means"].shape == (2, 100, k)
        else:
            assert trace.posterior["mixture/means"].shape == (2, 100, n, k)


def test_prior_predictive(mixture):
    model, n, _ = mixture
    ppc = pm.sample_prior_predictive(model, sample_shape=100).prior_predictive
    if n == 1:
        assert ppc["mixture/mixture"].shape == (1, 100)
    else:
        assert ppc["mixture/mixture"].shape == (1, 100, n)


def test_posterior_predictive(mixture):
    model, n, _ = mixture
    trace = pm.sample(model, num_samples=100, num_chains=2)
    ppc = pm.sample_posterior_predictive(model, trace).posterior_predictive
    if n == 1:
        assert ppc["mixture/mixture"].shape == (2, 100, 100)
    else:
        assert ppc["mixture/mixture"].shape == (2, 100, 100, n)
