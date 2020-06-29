"""PyMC4 test configuration."""
import pytest
import pymc4 as pm
import tensorflow as tf
from mock import Mock


@pytest.fixture(scope="function", autouse=True)
def tf_seed():
    tf.random.set_seed(37208)  # random.org
    yield


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def use_xla(request):
    return request.param == "XLA"


@pytest.fixture(scope="module", params=["auto_batch", "trust_manual_batching"], ids=str)
def use_auto_batching(request):
    return request.param == "auto_batch"


@pytest.fixture(scope="module", params=[1, 10, 100], ids=str)
def n_draws(request):
    return request.param


@pytest.fixture(scope="function")
def mock_biwrap_functools_call(monkeypatch):
    """Mock functools partial to test execution path of pm.model decorator when used
    in both the called and uncalled configuration"""
    _functools = Mock()

    def _partial(*args, **kwargs):
        raise Exception("Mocked functools partial")

    _functools.partial = _partial

    monkeypatch.setattr(pm.utils, "functools", _functools)

