"""
Tests for PyMC4 context managers

Tests function by replacing method with another that will raise
a custom exception, then checking that the correct exception has been raised.

In this manner we are able to verify code execution paths inside of PyMC4
"""

import pytest
from .. import random_variables, _template_contexts, model


def raise_exception(message):
    def _raise_exception(*args):
        raise Exception(message)

    return _raise_exception


def test_free_forward_context_add_variable(monkeypatch):
    """Test that add_variable is called in Free Forward context"""

    err_string = "Free Forward Context add_variable"

    monkeypatch.setattr(
        _template_contexts.FreeForwardContext, "add_variable", raise_exception(err_string)
    )

    with pytest.raises(Exception) as err:
        random_variables.Normal("test_normal", mu=0, sigma=1)
        assert err_string in str(err)


def test_free_forward_context_var_as_backend_tensor(monkeypatch):
    """Test that random variable initializes in Free Forward context"""

    err_string = "Free Forward Context var_as_backend_tensor"

    monkeypatch.setattr(
        _template_contexts.FreeForwardContext, "var_as_backend_tensor", raise_exception(err_string)
    )

    var = random_variables.Normal("test_normal", mu=0, sigma=1)
    with pytest.raises(Exception) as err:
        var.as_tensor()
        assert err_string in str(err)


def test_forward_context_add_variable(monkeypatch, tf_session):
    """Test that add_variable is called in Forward context"""

    err_string = "Forward Context add_variable"

    @model
    def test_model():
        random_variables.Normal("test_normal", mu=0, sigma=1)

    # Check that correct context is utilized
    monkeypatch.setattr(
        _template_contexts.ForwardContext, "add_variable", raise_exception(err_string)
    )

    with pytest.raises(Exception) as err:
        _model = test_model.configure()
        assert err_string in str(err)


def test_forward_context_var_as_backend_tensor(monkeypatch, tf_session):
    """Test that var_as_backend_tensor is called in Forward context"""

    err_string = "Forward Context var_as_backend_tensor"

    @model
    def test_model():
        random_variables.Normal("test_normal", mu=0, sigma=1)

    _model = test_model.configure()

    # Check that var has been added to correct context
    assert len(_model._forward_context.vars) == 1

    # Check that correct context is utilized
    monkeypatch.setattr(
        _template_contexts.ForwardContext, "var_as_backend_tensor", raise_exception(err_string)
    )
    with pytest.raises(Exception) as err:
        tf_session.run(_model.forward_sample())
        assert err_string in str(err)


@pytest.mark.skip("Unsure how to use InferenceContext")
def test_inference_context():
    """Test that random variable initializes in Inference Context"""
    raise NotImplementedError
