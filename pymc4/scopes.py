import threading


class Scope(object):
    """
    PyMC4 scopes intended to store useful information during forward pass of the model.
    This is intended to have more functionality rather than just name scoping.
    So this class should be a starting point for further development.

    The class absorbs any keyword arguments passed there. Accessing any attribute should return
    either None or the passed value by keyword. :func:`Scope.chain` will return all
    attributes for context, starting from the last one.

    Examples
    --------
    >>> with Scope(var=1):
    ...     with Scope(var=3):
    ...         print(list(Scope.chain("var")))
    [1, 3]
    """

    _leaf = object()
    context = threading.local()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    def __getattr__(self, item):
        return self.__dict__.get(item)

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.context, "stack"):
            cls.context.stack = []
        return cls.context.stack

    @classmethod
    def chain(cls, attr, *, leaf=_leaf, predicate=lambda _: True, drop_none=False):
        for c in cls.get_contexts():
            if predicate(c):
                val = getattr(c, attr)
                if drop_none and val is None:
                    continue
                else:
                    yield val
        if leaf is not cls._leaf:
            yield leaf

    @classmethod
    def variable_name(cls, name):
        """
        Generate PyMC4 variable name based on name scope we are currently in

        Parameters
        ----------
        name : str|None
            The desired target name for a variable, can be any, including None

        Returns
        -------
        str : scoped name

        Examples
        --------
        >>> with Scope(name="inner"):
        ...     print(Scope.variable_name("leaf"))
        inner/leaf
        >>> with Scope(name="inner"):
        ...     with Scope():
        ...         print(Scope.variable_name("leaf1"))
        inner/leaf1
        """
        return "/".join(map(str, cls.chain("name", leaf=name, drop_none=True)))

    def __repr__(self):
        return "Scope({})".format(self.__dict__)


def name_scope(name):
    return Scope(name=name)


variable_name = Scope.variable_name
