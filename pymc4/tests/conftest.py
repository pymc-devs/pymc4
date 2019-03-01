"""PyMC4 test configuration."""
import pytest
import tensorflow as tf


@pytest.fixture(scope="function")
def tf_session():

    tf.random.set_random_seed(37208)  # random.org
    sess = tf.Session()
    yield sess

    sess.close()
    tf.reset_default_graph()
