import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import edward2 as ed
from pymc4.util import interceptors, graph as graph_utils


class InputDistribution(tf.contrib.distributions.Deterministic):
    """
    detectable class for input
    is input <==> isinstance(rv.distribution, InputDistribution)
    """


def Input(name, shape, dtype=None):
    return ed.as_random_variable(InputDistribution(name=name, shape=shape, dtype=dtype))


class Model(object):
    def __init__(self, name=None, graph=None, session=None, **config):
        self.__dict__.update(config)
        self.name = name
        self._f = None
        self._variables = None
        self._observed = dict()
        if session is None:
            session = tf.Session(graph=graph)
        self.session = session

    @property
    def graph(self):
        return self.session.graph

    def define(self, f):
        self._f = f
        shapes_collector = interceptors.CollectShapes()
        def not_input_dependent(collector, rv):
            return len(graph_utils.inputs([rv])) == 0
        rv_collector = interceptors.CollectVariables(filter=not_input_dependent)
        with self.graph.as_default(), ed.interception(interceptors.Chain(shapes_collector, rv_collector)):
            f(self)
        self._variables = shapes_collector.result
        with self.session.as_default():
            test_vals = rv_collector

    def observe(self, **observations):
        self._observed = observations
        return self

    def reset(self):
        self._observed = dict()
        return self
