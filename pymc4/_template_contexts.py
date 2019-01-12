"""Contexts for model template evaluation.

When a new random variable is created, it will register
itself with the current context on the context stack by
calling the `add_variable` method of the context.

When a random variable is used in an arithmetic expression
it is converted to a backend (tensorflow) tensor using
`var_as_backend_tensor`.
"""
import threading


class BaseContext:
    """A context """

    def add_variable(self, rv):
        raise NotImplementedError("Abstract method.")

    def var_as_backend_tensor(self, rv):
        raise NotImplementedError("Abstract method.")

    def __enter__(self):
        _contexts.stack.append(self)
        return self

    def __exit__(self, *args):
        assert _contexts.stack[-1] is self
        _contexts.stack.pop()


class FreeForwardContext(BaseContext):
    def add_variable(self, rv):
        pass

    def var_as_backend_tensor(self, rv):
        return rv.sample()


class ForwardContext(BaseContext):
    def __init__(self):
        self.vars = []

    def add_variable(self, rv):
        self.vars.append(rv)

    def var_as_backend_tensor(self, rv):
        return rv.sample()


class InferenceContext(BaseContext):
    def __init__(self, tensors, expected_vars):
        self.vars = []
        self._tensors = {var.name: tensor for var, tensor in zip(expected_vars, tensors)}

    def add_variable(self, rv):
        self.vars.append(rv)

    def var_as_backend_tensor(self, rv):
        return self._tensors[rv.name]


_contexts = threading.local()
_contexts.stack = []
_contexts.default = FreeForwardContext()


def get_context():
    if len(_contexts.stack) == 0:
        return _contexts.default
    return _contexts.stack[-1]
