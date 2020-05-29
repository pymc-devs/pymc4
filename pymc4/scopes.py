"""Module that defines scope classes to easily create name scopes or any other kind of scope."""
from typing import Optional, List, Any, Callable, Generator, Union
import threading


class Scope(object):
    """
    General purpose variable scoping.

    PyMC4 scopes intended to store useful information during forward pass of the model.
    This is intended to have more functionality rather than just name scoping.
    So this class should be a starting point for further development.

    The class absorbs any keyword arguments passed there. Accessing any attribute should return
    either None or the passed value by keyword. :func:`Scope.chain` will return all
    attributes for context, starting from the first one (the deepest one is the last one).

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
        """Enter a new scope context appending it to the ``Scopes.context.stack``."""
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        """Remove the last scope context from the ``Scopes.context.stack``."""
        type(self).get_contexts().pop()

    def __getattr__(self, item):
        """Get a ``Scope`` instance's attribute ``item``."""
        return self.__dict__.get(item)

    @classmethod
    def get_contexts(cls) -> List:
        """Get the ``Scope`` class's context stack list.
        
        Returns
        -------
        contexts : List
            If this is made from outside of a ``with Scopes(...)`` statement,
            an empty list is returned. In any other case, it returns the
            nestedA ``list`` of ``Scope`` instances that are in the context of the
            
        """
        # no race-condition here, cls.context is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.context, "stack"):
            cls.context.stack = []
        return cls.context.stack

    @classmethod
    def chain(
        cls,
        attr: str,
        *,
        predicate: Callable[[Any], bool] = lambda _: True,
        drop_none: bool = False,
        leaf: Any = _leaf,
    ) -> Generator[Any, None, None]:
        """Yield all the values of a scoped attribute starting from the first to the deepest.

        Each ``Scope`` context manager can be used to add any given attribute's value to the context.
        This method explores the entire context stack an iterates through the values defined for a
        given attribute. It goes through the context stack from the first defined open context
        (the outer most scope) to the last ``Scope`` that was entered (inner most scope).

        Parameters
        ----------
        cls : Scope
            The ``Scope`` subclass.
        attr : str
            The name of the ``Scope`` attribute to get.
        predicate : Callable[[Any], bool]
            A function used to filter scope instances encountered in the context stack. Its signature
            must take a single input argument and return ``True`` or ``False``. If it returns
            ``True``, the ``Scope`` will be processed further. This means that the ``attr``
            attribute's value will be read from the ``Scope`` instance, and said value will be yielded
            (depending on ``drop_none``). If ``False``, the encountered scope will be skiped.
            By default, all encountered ``Scope`` instances are accepted for further processing.
        drop_none : bool
            If ``True`` and the ``attr`` value that is retrieved is ``None``, it is skipped.
            If ``False``, ``None`` will be yielded.
        leaf : Any
            A value to yield after having iterated through the entire context stack. By default,
            no value is yielded.

        Yields
        ------
        Any
            The values of the attribute ``attr`` that are defined in the context stack and
            optionally, the ``leaf`` value.

        Example
        -------
        If we nest several ``Scope`` instances, ``Scope.chain`` can iterate through the
        context stack looking for an attribute's value.

        >>> with Scope(var=1):
        ...     with Scope(var=3):
        ...         print(list(Scope.chain("var")))
        [1, 3]

        If one of the nested ``Scope`` instance doesn't define an attribute's value or defines
        it as ``None``, it is not included in the yielded values by default.

        >>> with Scope(var=1, name="A"):
        ...     with Scope(var=3):
        ...         print(list(Scope.chain("name", drop_none=True)))
        ['A']

        If we provide a ``leaf`` value, it will be returned as long as it isn't ``None`` and
        at the same time pass ``drop_none=True``.

        >>> with Scope(var=1, name="A"):
        ...     with Scope(var=3):
        ...         print(list(Scope.chain("name", leaf="leaf", drop_none=True)))
        ['A', 'leaf']
        
        """
        for c in cls.get_contexts():
            if predicate(c):
                val = getattr(c, attr)
                if drop_none and val is None:
                    continue
                else:
                    yield val
        if leaf is not cls._leaf:
            if not (drop_none and leaf is None):
                yield leaf

    @classmethod
    def variable_name(cls, name: Optional[str]) -> Optional[str]:
        """
        Generate PyMC4 variable name based on name scope we are currently in.

        Parameters
        ----------
        name : Union[str, None]
        The desired target name for a variable. If ``None``, it will simply
        return the chained scope's ``name`` attribute.

        Returns
        -------
        scoped_name : Union[str, None]
        If ``name`` is ``None`` and no scope defines the ``name`` attribute, this
        function returns ``None``.

        Examples
        --------
        >>> with Scope(name="inner"):
        ...     print(Scope.variable_name("leaf"))
        inner/leaf
        >>> with Scope(name="inner"):
        ...     with Scope():
        ...         print(Scope.variable_name("leaf1"))
        inner/leaf1

        empty name results in None name
        >>> assert Scope.variable_name(None) is None
        >>> assert Scope.variable_name("") is None
        """
        value = "/".join(map(str, cls.chain("name", leaf=name, drop_none=True)))
        if not value:
            return None
        else:
            return value

    @classmethod
    def transformed_variable_name(cls, transform_name: str, name: str) -> Optional[str]:
        """
        Generate PyMC4 transformed variable name based on name scope we are currently in.

        Parameters
        ----------
        transform_name : str
        The name of the transformation.
        name : str
        The plain name of the variable.

        Returns
        -------
        str : scoped name
        This is equivalent to calling :meth:`~.variable_name` with the input
        ``"__{transform_name}_{name}"``.

        Examples
        --------
        >>> with Scope(name="inner"):
        ...     print(Scope.variable_name("leaf"))
        inner/leaf
        >>> with Scope(name="inner"):
        ...     with Scope():
        ...         print(Scope.variable_name("leaf1"))
        inner/leaf1

        empty name results in None name
        >>> assert Scope.variable_name(None) is None
        >>> assert Scope.variable_name("") is None
        """
        return cls.variable_name("__{}_{}".format(transform_name, name))

    def __repr__(self) -> str:
        """Return the string representation of a ``Scope`` instance.

        Returns
        -------
        str:
            Returns ``"Scope({self.__dict__})"``.
        """
        return "Scope({})".format(self.__dict__)


def name_scope(name: Union[str, None]) -> Scope:
    """Create a :class:`~.Scope` instance with a "name" attribute and sets its value to the provided ``name``.

    Parameters
    ----------
    name : Union[str, None]
    The value that will be set to the ``Scope.name`` attribute.

    Returns
    -------
    scope : Scope
    A scope instance that only defines the ``name`` attribute.
    """
    return Scope(name=name)


variable_name = Scope.variable_name
transformed_variable_name = Scope.transformed_variable_name
