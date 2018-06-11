from tensorflow_probability import edward2 as ed
import pytest
import pymc4 as pm


def test_model_definition_type1():
    model = pm.Model()

    @model.define
    def simple(cfg):
        ed.Normal(0., 1., name='normal')

    assert 'normal' in model.variables
    assert [] == model.variables['normal'].shape.as_list()


def test_model_definition_type2():
    with pytest.raises(KeyError) as e:
        @pm.inline
        def model(cfg):
            ed.Normal(0., 1., name='normal', sample_shape=cfg.shape_for_normal)
    assert e.match('you probably need to pass "shape_for_normal" in model definition')

    @pm.inline(shape_for_normal=(10,))
    def model(cfg):
        ed.Normal(0., 1., name='normal', sample_shape=cfg.shape_for_normal)

    assert 'normal' in model.variables
    assert [10] == model.variables['normal'].shape.as_list()


def test_model_reconfigure():
    @pm.inline(shape_for_normal=(10,))
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
