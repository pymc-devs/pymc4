import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


@pytest.fixture(scope="module", params=[200])
def num_observed_samples(request):
    return request.param


@pytest.fixture(scope="module", params=[5000])
def replicas(request):
    return request.param


@pytest.fixture(scope="module", params=range(0, 3))
def batch_stack(request):
    ri = request.param
    return [1] * (ri + 1)


@pytest.fixture(scope="module", params=range(10, 12))
def simple_model(request, num_observed_samples, replicas):
    seed = request.param
    tf.random.set_seed(seed)
    mean = np.random.random()
    observed = tf.random.normal((num_observed_samples,)) + mean

    @pm.model
    def simple_model():
        pr = yield pm.Normal("pr", 0, 1)
        lkh = yield pm.Normal("lkh", 0, 1, observed=observed)
        return lkh

    return simple_model, mean


@pytest.fixture(scope="module", params=range(10, 12))
def model_conditioned(request, num_observed_samples):
    seed = request.param
    tf.random.set_seed(seed)
    mean = np.random.random()
    observed = tf.random.normal((num_observed_samples,)) + mean
    prior = "pr"

    @pm.model
    def model_conditioned():
        pr = yield pm.Normal(prior, 0, 1)
        lkh = yield pm.Normal("lkh", pr, 1, observed=observed)
        return lkh

    return model_conditioned, mean, prior


@pytest.fixture(scope="module", params=range(10, 12))
def model_batch_stack_prior(request, num_observed_samples, batch_stack):
    seed = request.param
    tf.random.set_seed(seed)
    mean = tf.random.normal((*batch_stack,))
    observed = tf.random.normal((num_observed_samples, *batch_stack)) + mean
    prior = "pr"

    @pm.model
    def model_conditioned():
        pr = yield pm.Normal(prior, 0, 1, batch_stack=batch_stack)
        lkh = yield pm.Normal("lkh", pr, 1, observed=observed)
        return lkh

    return model_conditioned, mean, prior


@pytest.fixture(scope="module", params=range(10, 12))
def model_batch_stack_lkh(request, num_observed_samples, batch_stack):
    seed = request.param
    tf.random.set_seed(seed)
    mean = tf.random.normal((*batch_stack,))
    observed = tf.random.normal((num_observed_samples, *batch_stack)) + mean
    prior = "pr"

    @pm.model
    def model_conditioned():
        pr = yield pm.Normal("pr", 0, 1)
        lkh = yield pm.Normal("lkh", pr, 1, batch_stack=batch_stack, observed=observed)
        return lkh

    return model_conditioned, mean, prior


@pytest.fixture(scope="module")
def model_no_observed(request):
    @pm.model
    def model_no_observed():
        pr = yield pm.Normal("pr", 0, 1)
        lkh = yield pm.Normal("lkh", 0, 1)
        return lkh

    return model_no_observed


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            "XLA",
            marks=pytest.mark.xfail(reason="XLA compilation in sample is not fully supported yet"),
        ),
        "noXLA",
    ],
    ids=str,
)
def xla_fixture(request):
    return request.param == "XLA"


def test_simple_model(simple_model, xla_fixture, replicas):
    model, mean = simple_model
    samples, map_ = pm.sample_smc(model(), replicas=replicas, xla=xla_fixture)


def test_model_batch_stack_prior(model_batch_stack_prior, xla_fixture, replicas):
    model, mean, prior = model_batch_stack_prior
    samples, map_ = pm.sample_smc(model(), replicas=replicas, xla=xla_fixture)
    mean_posterior = tf.reduce_mean(samples[0])
    np.testing.assert_allclose(mean_posterior, mean, rtol=5e-1)


def test_model_conditioned(model_batch_stack_lkh, xla_fixture, replicas):
    model, mean, prior = model_batch_stack_lkh
    samples, map_ = pm.sample_smc(model(), replicas=replicas, xla=xla_fixture)
    mean_posterior = tf.reduce_mean(samples[0], [0, 1])
    np.testing.assert_allclose(mean_posterior, mean, rtol=5e-1)


def test_model_conditioned(model_conditioned, xla_fixture, replicas):
    model, mean, prior = model_conditioned
    samples, map_ = pm.sample_smc(model(), replicas=replicas, xla=xla_fixture)
    mean_posterior = tf.reduce_mean(samples[0], [0, 1])
    np.testing.assert_allclose(mean_posterior, mean, rtol=5e-1)


def test_model_no_observed(model_no_observed):
    model = model_no_observed()
    with pytest.raises(ValueError):
        samples, map_ = pm.sample_smc(model)
