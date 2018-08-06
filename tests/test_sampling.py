from tensorflow_probability import edward2 as ed
import tensorflow as tf
import pytest
import pymc4 as pm


def test_sample():
    model = pm.Model()

    @model.define
    def sample(cfg):
        mu = ed.Normal(0., 1., name="mu")

    trace = pm.sample(model)

    assert 0. == pytest.approx(trace["mu"].mean(), 1)
    assert 1. == pytest.approx(trace["mu"].std(), 1)
