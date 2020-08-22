import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf
from pymc4.mcmc.samplers import RandomWalkM
from pymc4.mcmc.samplers import reg_samplers


@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        var1 = yield pm.Normal("var1", 0, 1)
        return var1

    return simple_model


@pytest.fixture(scope="function")
def compound_model():
    @pm.model()
    def compound_model():
        var1 = yield pm.Normal("var1", 0, 1)
        var2 = yield pm.Bernoulli("var2", 0.5)
        return var2

    return compound_model


@pytest.fixture(scope="function")
def categorical_same_shape():
    @pm.model
    def categorical_same_shape():
        var1 = yield pm.Categorical("var1", probs=[0.2, 0.4, 0.4])
        var1 = yield pm.Categorical("var2", probs=[0.1, 0.3, 0.6])
        var1 = yield pm.Categorical("var3", probs=[0.1, 0.1, 0.8])

    return categorical_same_shape


@pytest.fixture(scope="function")
def model_symmetric():
    @pm.model
    def model_symmetric():
        var1 = yield pm.Bernoulli("var1", 0.1)
        var2 = yield pm.Bernoulli("var2", 1 - 0.1)

    return model_symmetric


@pytest.fixture(scope="function")
def categorical_different_shape():
    @pm.model
    def categorical_different_shape():
        var1 = yield pm.Categorical("var1", probs=[0.2, 0.4, 0.4])
        var1 = yield pm.Categorical("var2", probs=[0.1, 0.3, 0.1, 0.5])
        var1 = yield pm.Categorical("var3", probs=[0.1, 0.1, 0.1, 0.2, 0.5])

    return categorical_different_shape


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


@pytest.fixture(scope="module", params=[3, 5])
def seed(request):
    return request.param


@pytest.fixture(scope="module", params=["hmc", "nuts", "rwm", "compound"])
def sampler_type(request):
    return request.param


@pytest.fixture(scope="module", params=["rwm", "compound"])
def discrete_support_sampler_type(request):
    return request.param


@pytest.fixture(scope="module", params=["nuts_simple", "hmc_simple"])
def expanded_sampler_type(request):
    return request.param


def test_samplers_on_compound_model(compound_model, seed, xla_fixture, sampler_type):
    def _execute():
        model = compound_model()
        trace = pm.sample(model, sampler_type=sampler_type, xla_fixture=xla_fixture, seed=seed)
        var1 = round(trace.posterior["compound_model/var1"].mean().item(), 1)
        # int32 dtype variable
        var2 = tf.reduce_sum(trace.posterior["compound_model/var2"]) / (1000 * 10)
        np.testing.assert_allclose(var1, 0.0, atol=0.1)
        np.testing.assert_allclose(var2, 0.5, atol=0.1)

    if sampler_type in ["compound", "rwm"]:
        # execute normally if sampler supports discrete distributions
        _execute()
    else:
        # else check for the exception thrown
        with pytest.raises(ValueError):
            _execute()


def test_compound_model_sampler_method(
    compound_model, seed, xla_fixture, discrete_support_sampler_type
):
    model = compound_model()
    trace = pm.sample(
        model,
        sampler_type=discrete_support_sampler_type,
        sampler_methods=[("var2", RandomWalkM)],
        xla_fixture=xla_fixture,
        seed=seed,
    )
    var1 = round(trace.posterior["compound_model/var1"].mean().item(), 1)
    # int32 dtype variable
    var2 = tf.reduce_sum(trace.posterior["compound_model/var2"]) / (1000 * 10)
    np.testing.assert_allclose(var1, 0.0, atol=0.1)
    np.testing.assert_allclose(var2, 0.5, atol=0.1)


def test_samplers_on_simple_model(simple_model, xla_fixture, sampler_type):
    model = simple_model()
    trace = pm.sample(model, sampler_type=sampler_type, xla_fixture=xla_fixture)
    var1 = round(trace.posterior["simple_model/var1"].mean().item(), 1)
    np.testing.assert_allclose(var1, 0.0, atol=0.1)


def test_extended_samplers_on_simple_model(simple_model, xla_fixture, expanded_sampler_type):
    model = simple_model()
    trace = pm.sample(model, sampler_type=expanded_sampler_type, xla_fixture=xla_fixture)
    var1 = round(trace.posterior["simple_model/var1"].mean().item(), 1)
    np.testing.assert_allclose(var1, 0.0, atol=0.1)


def test_simple_seed(simple_model, seed):
    model = simple_model()
    trace1 = pm.sample(model, xla_fixture=xla_fixture, seed=seed)
    trace2 = pm.sample(model, xla_fixture=xla_fixture, seed=seed)
    np.testing.assert_allclose(
        tf.norm(trace1.posterior["simple_model/var1"] - trace2.posterior["simple_model/var1"]),
        0.0,
        atol=1e-6,
    )


def test_compound_seed(compound_model, seed):
    model = compound_model()
    trace1 = pm.sample(model, xla_fixture=xla_fixture, seed=seed)
    trace2 = pm.sample(model, xla_fixture=xla_fixture, seed=seed)
    np.testing.assert_allclose(
        tf.norm(
            tf.cast(
                trace1.posterior["compound_model/var1"] - trace2.posterior["compound_model/var1"],
                dtype=tf.float32,
            )
        ),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tf.norm(
            tf.cast(
                trace1.posterior["compound_model/var2"] - trace2.posterior["compound_model/var2"],
                dtype=tf.float32,
            )
        ),
        0.0,
        atol=1e-6,
    )


def test_sampler_merging(categorical_same_shape, categorical_different_shape):
    model_same = categorical_same_shape()
    model_diff = categorical_different_shape()
    sampler = reg_samplers["compound"]
    sampler1 = sampler(model_same)
    sampler1._assign_default_methods()
    sampler2 = sampler(model_diff)
    sampler2._assign_default_methods()
    assert len(sampler1.kernel_kwargs["compound_samplers"]) == 1
    assert len(sampler2.kernel_kwargs["compound_samplers"]) == 3
    sampler_methods1 = [("var1", RandomWalkM)]
    sampler_methods2 = [
        ("var1", RandomWalkM, {"new_state_fn": pm.categorical_uniform_fn(classes=3)})
    ]
    sampler_methods3 = [
        (
            "var1",
            RandomWalkM,
            {"new_state_fn": pm.categorical_uniform_fn(classes=3, name="smth_different")},
        )
    ]

    sampler_methods4 = [
        (
            "var1",
            RandomWalkM,
            {"new_state_fn": pm.categorical_uniform_fn(classes=3, name="smth_different")},
        ),
        (
            "var3",
            RandomWalkM,
            {"new_state_fn": pm.categorical_uniform_fn(classes=3, name="smth_different")},
        ),
    ]

    sampler_methods5 = [
        ("var1", RandomWalkM,),
        ("var2", RandomWalkM,),
        ("var3", RandomWalkM,),
    ]

    sampler_ = sampler(model_same)
    sampler_._assign_default_methods(sampler_methods=sampler_methods1)
    assert len(sampler_.kernel_kwargs["compound_samplers"]) == 1
    sampler_._assign_default_methods(sampler_methods=sampler_methods2)
    assert len(sampler_.kernel_kwargs["compound_samplers"]) == 1
    sampler_._assign_default_methods(sampler_methods=sampler_methods3)
    assert len(sampler_.kernel_kwargs["compound_samplers"]) == 2
    sampler_._assign_default_methods(sampler_methods=sampler_methods4)
    assert len(sampler_.kernel_kwargs["compound_samplers"]) == 2
    sampler_ = sampler(model_diff)
    sampler_._assign_default_methods(sampler_methods=sampler_methods5)
    assert len(sampler_.kernel_kwargs["compound_samplers"]) == 3


def test_other_samplers(simple_model, seed):
    model = simple_model()
    trace1 = pm.sample(model, sampler_type="nuts_simple", xla_fixture=xla_fixture, seed=seed)
    trace2 = pm.sample(model, sampler_type="hmc_simple", xla_fixture=xla_fixture, seed=seed)
    np.testing.assert_allclose(tf.reduce_mean(trace1.posterior["simple_model/var1"]), 0.0, atol=0.1)
    np.testing.assert_allclose(tf.reduce_mean(trace2.posterior["simple_model/var1"]), 0.0, atol=0.1)


def test_compound_symmetric(model_symmetric, seed):
    model = model_symmetric()
    trace = pm.sample(model)
    np.testing.assert_allclose(
        tf.reduce_mean(tf.cast(trace.posterior["model_symmetric/var1"], dtype=tf.float32)),
        0.1,
        atol=0.1,
    )
    np.testing.assert_allclose(
        tf.reduce_mean(tf.cast(trace.posterior["model_symmetric/var2"], dtype=tf.float32)),
        1.0 - 0.1,
        atol=0.1,
    )
