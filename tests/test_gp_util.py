import tensorflow as tf
import numpy as np
import pymc4 as pm

import pytest

doc_string = "Func doc"


def test_stabilize_default_shift():
    data = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    shifted = pm.gp.util.stabilize(data)
    expected = tf.constant([[1.0001, 2.0], [3.0, 4.0001]])
    assert np.allclose(shifted, expected, rtol=1e-18)
    data = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float64)
    shifted = pm.gp.util.stabilize(data)
    expected = tf.constant([[1.000001, 2.0], [3.0, 4.000001]])


def test_stabilize():
    data = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    shifted = pm.gp.util.stabilize(data, shift=1.0)
    expected = tf.constant([[2.0, 2.0], [3.0, 5.0]])
    assert np.allclose(shifted, expected, rtol=1e-18)


def test_inherit_docs():
    def func():
        """
        Func docs.
        """
        pass

    @pm.gp.util._inherit_docs(func)
    def other_func():
        pass

    assert other_func.__doc__ == func.__doc__


def test_inherit_docs_exception():
    def func():
        pass

    with pytest.raises(ValueError, match=r"No docs to inherit"):

        @pm.gp.util._inherit_docs(func)
        def other_func():
            pass


def test_build_docs():
    @pm.gp.util._build_docs
    def func():
        """%(doc_string)"""
        pass

    assert func.__doc__ == doc_string


def test_build_doc_warning():
    with pytest.warns(SyntaxWarning, match=r"arrtibute nodoc not found"):

        @pm.gp.util._build_docs
        def func():
            """%(nodoc)"""
            pass
