"""
Test PyMC4 Utils
"""

import pymc4 as pm
import pytest
from mock import Mock


@pytest.fixture(scope="function")
def mock_biwrap_functools_call(monkeypatch):
    """Mock functools partial to test execution path of pm.model decorator when used
    in both the called and uncalled configuration"""
    _functools = Mock()

    def _partial(*args, **kwargs):
        raise Exception("Mocked functools partial")

    _functools.partial = _partial

    monkeypatch.setattr(pm.utils, "functools", _functools)


def test_biwrao_and_mocked_functools_raises_exception_with_called_decorator(
    mock_biwrap_functools_call
):
    """Test code path for called decorator by adding exception to to pm4.utils.functools.partial"""

    with pytest.raises(Exception) as e:

        @pm.model()
        def fake_model():
            yield None

        assert "Mocked functools partial" in str(e)


def test_biwrap_with_uncalled_decorator(mock_biwrap_functools_call):
    """Test code path not taken by verifying exception is not raised by pm.utils.functools.partial"""
    with pytest.raises(Exception) as e:
        # Verify that functools.partial has been mocked correctly.
        # If this section is failing then test suite is configured in correctly
        pm.utils.functools.partial()
        assert "Mocked functools partial" in str(e)

    @pm.model
    def fake_model():
        yield None
