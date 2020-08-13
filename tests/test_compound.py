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


@pytest.fixture(scope="module", params=[3, 5, 7])
def seed(request):
    return request.param


@pytest.fixture(scope="module", params=["hmc", "nuts", "randomwalkm", "compound"])
def sampler_type(request):
    return request.param


@pytest.fixture(scope="module", params=["randomwalkm", "compound"])
def discrete_support_sampler_type(request):
    return request.param


@pytest.fixture(scope="module", params=["nuts_simple", "hmc_simple"])
def expanded_sampler_type(request):
    return request.param


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

    sampler_methods5 = [("var1", RandomWalkM,), ("var2", RandomWalkM,), ("var3", RandomWalkM,)]

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
