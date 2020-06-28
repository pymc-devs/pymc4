import itertools
import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


@pytest.fixture(scope="module", params=[200, 300])
def num_observed_samples(request):
    return request.param


@pytest.fixture(scope="module", params=[2000, 5000])
def draws(request):
    return request.param


@pytest.fixture(scope="module", params=range(0, 3))
def batch_stack(request):
    ri = request.param
    return [1] * (ri + 1)


@pytest.fixture(scope="module", params=range(10, 13))
def simple_model(request, num_observed_samples, draws):
    seed = request.param
    tf.random.set_seed(seed)
    observed = tf.random.normal((num_observed_samples,))

    @pm.model
    def simple_model():
        pr = yield pm.Normal("pr", 0, 1)
        lkh = yield pm.Normal("lkh", 0, 1, observed=observed)
        return lkh

    return simple_model


@pytest.fixture(scope="module", params=range(10, 13))
def model_conditioned(request, num_observed_samples, draws):
    seed = request.param
    tf.random.set_seed(seed)
    observed = tf.random.normal((num_observed_samples,))

    @pm.model
    def model_conditioned():
        pr = yield pm.Normal("pr", 0, 1)
        lkh = yield pm.Normal("lkh", pr, 1, observed=observed)
        return lkh

    return model_conditioned


@pytest.fixture(scope="module", params=range(10, 13))
def model_batch_stack_prior(request, num_observed_samples, draws, batch_stack):
    seed = request.param
    tf.random.set_seed(seed)
    observed = tf.random.normal((num_observed_samples,))

    @pm.model
    def model_conditioned():
        pr = yield pm.Normal("pr", 0, 1, batch_stack=batch_stack)
        lkh = yield pm.Normal("lkh", pr, 1, observed=observed)
        return lkh

    return model_conditioned


@pytest.fixture(scope="module", params=range(10, 13))
def model_batch_stack_lkh(request, num_observed_samples, draws, batch_stack):
    seed = request.param
    tf.random.set_seed(seed)
    observed = tf.random.normal((num_observed_samples,))

    @pm.model
    def model_conditioned():
        pr = yield pm.Normal("pr", 0, 1)
        lkh = yield pm.Normal("lkh", pr, 1, batch_stack=batch_stack, observed=observed)
        return lkh

    return model_conditioned


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


def test_simple_model(simple_model, xla_fixture, draws):
    model = simple_model()
    samples, map_ = pm.sample_smc(model, draws=draws)


# TODO, should think more on the comprehensive testing
