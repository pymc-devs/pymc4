import biwrap
import tensorflow as tf
from tensorflow_probability import edward2 as ed
from pymc4.util import interceptors

__all__ = ['Model', 'inline']


class Config(dict):
    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            error = KeyError(item, '"{i}" is not found in configuration for the model, '
                             'you probably need to pass "{i}" in model definition as '
                             '\n`model = pm.Model(..., {i}=value)`'
                             '\nor'
                             '\n'
                             '\n@pm.inline(..., {i}=value)'
                             '\ndef model(cfg):'
                             '\n    # your model starts here'
                             '\n    ...'.format(i=item))
            raise error from e


class Model(object):
    def __init__(self, name=None, graph=None, session=None, **config):
        self._cfg = Config(**config)
        self.name = name
        self._f = None
        self._variables = None
        self._observed = dict()
        if session is None:
            session = tf.Session(graph=graph)
        self.session = session

    def define(self, f):
        self._f = f
        self._init_variables()
        return f

    def configure(self, **override):
        self._cfg.update(**override)
        self._init_variables()
        return self

    def _init_variables(self):
        info_collector = interceptors.CollectVariablesInfo()
        with self.graph.as_default(), ed.interception(info_collector):
            self._f(self.cfg)
        self._variables = info_collector.result

    def test_point(self, sample=True):
        def not_observed(var, *args, **kwargs):
            return kwargs['name'] not in self.observed
        values_collector = interceptors.CollectVariables(filter=not_observed)
        chain = [values_collector]
        if not sample:

            def get_mode(state, rv, *args, **kwargs):
                return rv.distribution.mode()
            chain.insert(0, interceptors.Generic(after=get_mode))

        with self.graph.as_default(), ed.interception(interceptors.Chain(*chain)):
            self._f(self.cfg)
        with self.session.as_default():
            returns = self.session.run(list(values_collector.result.values()))
        return dict(zip(values_collector.result.keys(), returns))

    def observe(self, **observations):
        self._observed = observations
        return self

    def reset(self):
        self._observed = dict()
        return self

    @property
    def graph(self):
        return self.session.graph

    @property
    def observed(self):
        return self._observed

    @property
    def variables(self):
        return self._variables

    @property
    def cfg(self):
        return self._cfg


@biwrap.biwrap
def inline(f, **kwargs):
    model = Model(**kwargs)
    model.define(f)
    return model
