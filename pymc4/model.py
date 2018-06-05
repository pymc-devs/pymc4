import threading

__all__ = ["Model", "modelcontext"]


class Context(object): # pylint: disable=too-few-public-methods
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """
    contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()


class Model(Context):
    def __new__(cls, **kwargs):
        instance = super(Model, cls).__new__(cls)
        if kwargs.get('model') is not None:
            instance.parent = kwargs.get('model')
        elif cls.get_contexts():
            instance.parent = cls.get_contexts()[-1]
        else:
            instance.parent = None
        return instance

    def __init__(self, name="", model=None, ):
        self.name = name
        self.named_vars = {}
        self.parent = model

    @property
    def model(self):
        return self

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("This must be called from inside a Model context manager")

    def add_random_variable(self, var):
        """Add a random variable to the named variables of the model."""
        if var.name in self.named_vars:
            raise ValueError(
                "Variable name {} already exists.".format(var.name))
        self.named_vars[var.name] = var


def modelcontext(model):
    """return the given model or try to find it in the context if there was
    none supplied.
    """
    if model is None:
        return Model.get_context()
    return model
