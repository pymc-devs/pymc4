"""
PyMC4 test configuration
"""
import pytest
import tensorflow as tf


@pytest.fixture(scope="function")
def tf_session(request):
    sess = tf.Session()

    def fin():
        tf.reset_default_graph()

    request.addfinalizer(fin)
    
    return sess
