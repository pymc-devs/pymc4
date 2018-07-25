from tensorflow_probability import edward2 as ed
import tensorflow as tf
import pytest
import pymc4 as pm
from pymc4.model.base import Config
# pylint: disable=unused-variable, unused-argument
# pylint: disable-msg=E0102


def test_model_definition_type1():
    model = pm.Model(name="testName")

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    assert 'normal' in model.variables
    assert [] == model.variables['normal'].shape.as_list()
    assert model.name == "testName"


def test_model_definition_type2():
    with pytest.raises(KeyError) as e:
        @pm.inline
        def model(cfg):
            ed.Normal(0., 1., name='normal', sample_shape=cfg.shape_for_normal)
    assert e.match('you probably need to pass "shape_for_normal" in model definition')

    @pm.inline(shape_for_normal=(10,))  # pylint: disable-msg=E1120
    def model(cfg):
        ed.Normal(0., 1., name='normal', sample_shape=cfg.shape_for_normal)

    assert 'normal' in model.variables
    assert [10] == model.variables['normal'].shape.as_list()


def test_model_reconfigure():
    @pm.inline(shape_for_normal=(10,))  # pylint: disable-msg=E1120
    def model(cfg):
        ed.Normal(0., 1., name='normal', sample_shape=cfg.shape_for_normal)

    assert 'normal' in model.variables
    assert [10] == model.variables['normal'].shape.as_list()
    model.configure(shape_for_normal=3)
    assert [3] == model.variables['normal'].shape.as_list()


def test_testvalue():
    @pm.inline
    def model(cfg):
        ed.Normal(0., 1., name='normal')

    testval_random = model.test_point()
    testval_mode = model.test_point(sample=False)
    assert testval_mode['normal'] == 0.
    assert testval_mode['normal'] != testval_random['normal']


def test_variables():
    model = pm.Model()

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    assert len(model.variables) == 1
    assert len(model.unobserved) == 1
    assert "normal" in model.variables


def test_model_target_log_prob_fn():
    model = pm.Model()

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    model.target_log_prob_fn()


def test_model_observe():

    model = pm.Model()

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    model.observe(normal=1)

    assert len(model.observed) == 1
    assert not model.unobserved


def test_model_reset():
    model = pm.Model()

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    model.observe(normal=1)

    assert len(model.observed) == 1
    assert not model.unobserved

    model.reset()

    assert not model.observed
    assert len(model.unobserved) == 1


def test_model_session():
    model = pm.Model()

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    assert isinstance(model.session, tf.Session)


def test_model_config():
    model = pm.Model()

    assert model.cfg == {}

    model = pm.Model(var1=123)

    @model.define
    def simple(cfg):
        assert cfg["var1"] == 123

    model = pm.Model(var1=123)

    @model.define
    def simple(cfg):
        pass

    model = model.configure(var1=12)

    @model.define
    def simple(cfg):
        assert cfg["var1"] == 12


def test_model_log_prob_fn():
    model = pm.Model()

    @model.define
    def simple(cfg):
        mu = ed.Normal(0., 1., name="mu")

    log_prob_fn = model.target_log_prob_fn()

    with tf.Session(graph=model.temp_graph):
        assert -0.91893853 == pytest.approx(log_prob_fn(0).eval(), 0.00001)
