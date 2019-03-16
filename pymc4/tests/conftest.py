"""PyMC4 test configuration."""
import pytest
import tensorflow as tf


@pytest.fixture(scope="function")
def tf_session():

    # TODO: Figure out how to set random seed in tf2
    # tf.random.set_random_seed(37208)  # random.org

    # TODO: Yield default eager execution graph for now
    # TODO: Figure out how to clear or namespace graph at later point

    # graph = tf.Graph()
    # with graph.as_default():
    # yield graph
    pass
