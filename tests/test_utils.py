"""
Test PyMC4 Utils
"""

import pymc4 as pm
import pytest


def test_biwrap_and_mocked_functools_raises_exception_with_called_decorator(
    mock_biwrap_functools_call,
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
