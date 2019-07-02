"""PyMC4 test configuration."""
import pytest
import tensorflow as tf


@pytest.fixture(scope="function")
def tf_seed():
    tf.random.set_seed(37208)  # random.org
