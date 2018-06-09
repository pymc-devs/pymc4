import collections
import tensorflow as tf

__all__ = [
    'SetState',
    'CollectShapes',
    'CollectVariables',
    'Chain',
    'CollectLogProb'
]


class Interceptor(object):
    def name_scope(self):
        return tf.name_scope(self.__class__.__name__.lower())

    def __call__(self, f, *args, **kwargs):
        if kwargs.get('name') is None:
            raise SyntaxError('Every random variable should have a name')
        f, args, kwargs = self.before(f, *args, **kwargs)
        observed = kwargs.pop('observed', False)
        rv = f(*args, **kwargs)
        kwargs['observed'] = observed
        return self.after(rv, *args, **kwargs)

    def before(self, f, *args, **kwargs):
        return f, args, kwargs

    def after(self, rv, *args, **kwargs):
        return rv


class Chain(Interceptor):
    def __init__(self, *interceptors, order_before=None, order_after=None):
        if order_before is None:
            order_before = tuple(range(len(interceptors)))
        if order_after is None:
            order_after = tuple(range(len(interceptors)))
        if len(order_before) != len(interceptors):
            raise ValueError('order_before should have the same length as interceptors or None')
        if len(order_after) != len(interceptors):
            raise ValueError('order_after should have the same length as interceptors or None')
        self._order_before = order_before
        self._order_after = order_after
        self._interceptors = interceptors

    def before(self, f, *args, **kwargs):
        for inperceptor in self._interceptors:
            f, args, kwargs = inperceptor.before(f, *args, **kwargs)
        return f, args, kwargs

    def after(self, rv, *args, **kwargs):
        for inperceptor in self._interceptors:
            rv = inperceptor.after(rv, *args, **kwargs)
        return rv


class SetState(Interceptor):
    def __init__(self, state):
        self.state = state

    def before(self, f, *args, **kwargs):
        if kwargs['name'] in self.state:
            kwargs['value'] = self.state[kwargs['name']]
        return f, args, kwargs


class CollectShapes(Interceptor):
    def __init__(self):
        self.result = collections.OrderedDict()

    def after(self, rv, *args, **kwargs):
        name = kwargs.get("name")
        if name not in self.result:
            self.result[name] = rv.shape
        else:
            raise KeyError(name, 'Duplicate name')
        return rv


class CollectVariables(Interceptor):
    def __init__(self, filter=None):
        self.filter = filter
        self.result = collections.OrderedDict()

    def after(self, rv, *args, **kwargs):
        if self.filter is not None and self.filter(self, rv):
            return rv
        name = kwargs.get("name")
        if name not in self.result:
            self.result[name] = rv
        else:
            raise KeyError(name, 'Duplicate name')
        return rv


class CollectLogProb(SetState):
    def __init__(self, state):
        super().__init__(state)
        with self.name_scope():
            self._result = tf.constant(0.)

    def before(self, f, *args, **kwargs):
        if kwargs['name'] not in self.state:
            raise RuntimeError(kwargs.get('name'), 'All RV should be present in state dict')
        return super().before(f, *args, **kwargs)

    def after(self, rv, *args, **kwargs):
        with self.name_scope():
            log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
            self._result += log_prob

    @property
    def result(self):
        with self.name_scope():
            return tf.identity(self._result, 'result')
