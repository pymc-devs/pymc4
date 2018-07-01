import collections
import biwrap
import tensorflow as tf
from tensorflow_probability import edward2 as ed
from pymc4.util import interceptors

__all__ = ['Model', 'inline']


class Config(dict):
    """
    Super class over dict class. Gives an error when a particular attribute does not exist.
    """
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
        self.observe(**config)

    def define(self, f):
        self._f = f
        self._init_variables()
        return f

    def configure(self, **override):
        self._cfg.update(**override)
        self._init_variables()
        self.observe(**self._cfg)
        return self

    def _init_variables(self):
        info_collector = interceptors.CollectVariablesInfo()
        with self.graph.as_default(), ed.interception(info_collector):
            self._f(self.cfg)
        self._variables = info_collector.result

    def test_point(self, sample=True):
        def not_observed(var, *args, **kwargs):  # pylint: disable=unused-argument
            return kwargs['name'] not in self.observed
        values_collector = interceptors.CollectVariables(filter=not_observed)
        chain = [values_collector]
        if not sample:

            def get_mode(state, rv, *args, **kwargs):  # pylint: disable=unused-argument
                return rv.distribution.mode()
            chain.insert(0, interceptors.Generic(after=get_mode))

        with self.graph.as_default(), ed.interception(interceptors.Chain(*chain)):
            self._f(self.cfg)
        with self.session.as_default():
            returns = self.session.run(list(values_collector.result.values()))
        return dict(zip(values_collector.result.keys(), returns))

    def target_log_prob_fn(self, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Pass the states of the RVs as args in alphabetical order of the RVs.
        Compatible as `target_log_prob_fn` for tfp samplers.
        """

        def log_joint_fn(*args, **kwargs):  # pylint: disable=unused-argument
            states = dict(zip(self.unobserved.keys(), args))
            states.update(self.observed)
            log_probs = []

            def interceptor(f, *args, **kwargs):
                name = kwargs.get("name")
                for name in states:
                    value = states[name]
                    if kwargs.get("name") == name:
                        kwargs["value"] = value

                rv = f(*args, **kwargs)
                log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
                log_probs.append(log_prob)
                return rv

            with ed.interception(interceptor):
                self._f(self._cfg)

            log_prob = sum(log_probs)
            return log_prob
        return log_joint_fn

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
    def unobserved(self):
        unobserved = collections.OrderedDict()
        for name, variable in self.variables.items():
            if name not in self.observed:
                unobserved[name] = variable

        return unobserved

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
