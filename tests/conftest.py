"""PyMC4 test configuration."""
import pytest
import tensorflow as tf


@pytest.fixture(scope="function", autouse=True)
def tf_seed():
    tf.random.set_seed(37208)  # random.org
    yield

