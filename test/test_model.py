import pytest
from tensorflow_probability import distributions as tfd
import pymc4 as pm
from pymc4.random_variable import RandomVariable as RV


class NewModel(pm.Model):
    def __init__(self, name='', model=None):
        super(NewModel, self).__init__(name, model)
        assert pm.modelcontext(self) is self
        with self:
            self._v1 = RV("v1", tfd.Normal(0., 1.))


class TestBaseModel(object):
    def test_no_of_named_vars(self):
        with pm.Model() as model1:
            RV("RV1", tfd.Normal(0., 1.))
            assert len(model1.named_vars) == 1
            with pm.Model() as model2:
                RV("RV2", tfd.Normal(0., 1.))
        assert len(model1.named_vars) == 2
        assert len(model2.named_vars) == 1

    def test_context_passes_vars_to_parent_model(self):
        with pm.Model() as model:
            # a set of variables is created
            NewModel(name="newmodel")
            # another set of variables are created but with prefix 'another'
            usermodel2 = pm.Model(name='another')
            # you can enter in a context with submodel
            with usermodel2:
                RV("v2", tfd.Normal(0., 1.))

                # this variable is created in parent model too
        assert 'another_v2' in model.named_vars
        #TODO: Add RV access thorugh as model attribute.

class TestNested(object):
    def test_nest_context_works(self):
        with pm.Model() as _m:
            new = NewModel()
            with new:
                assert pm.modelcontext(None) is new
            assert pm.modelcontext(None) is _m
            assert 'v1' in _m.named_vars

    def test_parent_attribute(self):
        model = pm.Model()
        with pm.Model(model=model) as model1 :
            assert model1.parent == model
def test_named_context():
    with pm.Model() as _m:
        NewModel(name='new')
    assert 'new_v1' in _m.named_vars

def test_in_model_context_error():
    with pytest.raises(TypeError):
        RV("RV", tfd.Normal(0., 1.))

def test_variable_name_exists_error():
    with pytest.raises(ValueError):
        with pm.Model() as _:
            RV("RV", tfd.Normal(0., 1.))
            RV("RV", tfd.Normal(0., 1.))

def test_model_property():
    with pm.Model() as model:
        assert model.model == model

def test_get_contexts():
    with pm.Model() as model:
        RV("RV", tfd.Normal(0., 1.))
        assert len(model.get_contexts()) == 1
        assert model.get_context() == model
        with pm.Model() as model1:
            RV("RV1", tfd.Normal(0., 1.))
            assert len(model.get_contexts()) == 2
            assert model1.get_context() == model1

def test_modelcontext():
    with pm.Model() as model:
        assert model.get_context() == model
        assert pm.modelcontext(model) == model
