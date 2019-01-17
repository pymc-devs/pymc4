"""
PyMC4 test configuration
"""
import pytest
import tensorflow as tf


@pytest.fixture(scope="function")
def tf_session():
    sess = tf.Session()
    yield sess

    tf.reset_default_graph()


