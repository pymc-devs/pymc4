"""
Tests for PyMC4 context managers
"""

import pytest
from .. import random_variables, _template_contexts, model


def raise_exception(message):
    def _raise_exception(*args):
        raise Exception(message)

    return _raise_exception


def test_free_forward_context(monkeypatch):
    """Test that random variable initializes with in Free Forward context"""

    monkeypatch.setattr(
        _template_contexts.FreeForwardContext,
        "var_as_backend_tensor",
        raise_exception("Free Forward Context"),
    )

    var = random_variables.Normal("test_normal", loc=0, scale=1)
    with pytest.raises(Exception) as err:
        var.as_tensor()
        assert "Free Forward Context" in str(err)


def test_forward_context(monkeypatch, tf_session):
    """Test that random variable initializes in Forward Context"""

    @model
    def test_model():
        random_variables.Normal("test_normal", loc=0, scale=1)

    _model = test_model.configure()

    # Check that var has been added to correct context
    assert len(_model._forward_context.vars) == 1

    # Check that correct context is utilized
    monkeypatch.setattr(
        _template_contexts.ForwardContext,
        "var_as_backend_tensor",
        raise_exception("Forward Context"),
    )
    with pytest.raises(Exception) as err:
        tf_session.run(_model.forward_sample())
        assert "Free Context" in str(err)


@pytest.mark.skip("Unsure how to use InferenceContext")
def test_inference_context():
    """Test that random variable initializes in Inference Context"""
    raise NotImplementedError
