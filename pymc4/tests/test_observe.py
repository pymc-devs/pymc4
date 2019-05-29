import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()
import tensorflow_probability as tfp

tfd = tfp.distributions
import pymc4 as pm
import pytest
import numpy as np


@pytest.fixture(scope="function")
def simple_model():
    @pm.model
    def make_model():
        x = pm.Normal(mu=0, sigma=1, name="x")
        y = pm.Normal(mu=x, sigma=1, name="y")

    return make_model.configure()


def true_joint_lp(x, y):
    return tf.reduce_sum(tfd.Normal(0, 1).log_prob(x)) + tf.reduce_sum(tfd.Normal(x, 1).log_prob(y))


def test_condition_x(simple_model):
    mod = simple_model
    mod = mod.observe(x=np.float32([1, 1, 1]))

    logp_func = mod.make_log_prob_function()
    logp_y = logp_func(y=np.float32([1, 2, 3]))
    assert tf.math.equal(logp_y, true_joint_lp(x=np.float32([1, 1, 1]), y=np.float32([1, 2, 3])))


def y(simple_model):
    mod = simple_model
    mod = mod.observe(y=np.float32([1, 2, 3]))

    logp_func = mod.make_log_prob_function()
    logp_x = logp_func(x=np.float32([1, 1, 1]))
    assert logp_x == pytest.approx(true_joint_lp(x=np.float32([1, 1, 1]), y=np.float32([1, 2, 3])))


def test_condition_xy(simple_model):
    mod = simple_model
    mod = mod.observe(x=np.float32([1, 1, 1]), y=np.float32([1, 2, 3]))

    logp_func = mod.make_log_prob_function()
    logp_xy = logp_func()
    assert tf.math.equal(logp_xy, true_joint_lp(x=np.float32([1, 1, 1]), y=np.float32([1, 2, 3])))
