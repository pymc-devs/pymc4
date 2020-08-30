"""Miscellaneous utility functions."""
import functools
import re
from typing import Callable, Sequence, Optional
import io
import pkgutil
import os


def biwrap(wrapper) -> Callable:  # noqa
    """Allow for optional keyword arguments in lower level decoratrors.

    Notes
    -----
    Currently this is only used to wrap pm.Model to capture model runtime flags such as
    keep_auxiliary and keep_return. See pm.Model for all possible keyword parameters

    """

    @functools.wraps(wrapper)
    def enhanced(*args, **kwargs) -> Callable:

        # Check if decorated method is bound to a class
        is_bound_method = hasattr(args[0], wrapper.__name__) if args else False
        if is_bound_method:
            # If bound to a class, `self` will be an argument
            count = 1
        else:
            count = 0
        if len(args) > count:
            # If lower level decorator is not called user model will be an argument
            # fill in parameters and call pm.Model
            newfn = wrapper(*args, **kwargs)
            return newfn
        else:
            # If lower level decorator is called user model will not be passed in as an argument
            # prefill args and kwargs but do not call pm.Model
            newwrapper = functools.partial(wrapper, *args, **kwargs)
            return newwrapper

    return enhanced


class NameParts:
    """Class that store names segmented into its three parts.

    A given name is made up by three parts:
    1. ``path``: usually represents the context or scope under which a name
    was defined. For example, a distribution inside a model will have its
    name's path equal to the model's full name. To conveniently store nested
    contexts, the path is usually stored as a tuple of strings.
    2. ``transform``: the name of the transformation that is applied to a
    distribution (can be an empty string, meaning no transformation).
    3. ``untransformed_name``: the name, stripped from its path and transform.
    This represents, for example, a distribution's plain name, without the
    its model's scope or any transformation name.
    """

    NAME_RE = re.compile(r"^(?:__(?P<transform>[^_]+)_)?(?P<name>[^_].*)$")
    NAME_ERROR_MESSAGE = (
        "Invalid name: `{}`, the correct one should look like: `__transform_name` or `name`, "
        "note only one underscore between the transform and actual name"
    )
    UNTRANSFORMED_NAME_ERROR_MESSAGE = (
        "Invalid name: `{}`, the correct one should look like: " "`name` without leading underscore"
    )
    __slots__ = ("path", "transform_name", "untransformed_name")

    @classmethod
    def is_valid_untransformed_name(cls, name: str) -> bool:
        """Test if a name can be used as an untransformed random variable.

        This function attempts to test if the supplied name, by accident,
        matches the naming pattern used for auto transformed random variables.
        If it does not, it is assumed to be a potentially valid name.

        Parameters
        ----------
        name : str
            The name to test.

        Returns
        -------
        bool
            ``False`` if the ``name`` matches the pattern used to give names
            to auto transformed variables. ``True`` otherwise.

        """
        match = cls.NAME_RE.match(name)
        return match is not None and match["transform"] is None

    @classmethod
    def is_valid_name(cls, name: str) -> bool:
        """Test if a name doesn't contain forbidden symbols.

        Parameters
        ----------
        name : str
            The name to test.

        Returns
        -------
        bool
            ``True`` if the ``name`` doesn't have forbidden symbols, ``False``
            otherwise.

        """
        match = cls.NAME_RE.match(name)
        return match is not None

    def __init__(
        self, path: Sequence[str], transform_name: Optional[str], untransformed_name: str,
    ):
        """Initialize a ``NameParts`` instance from its parts.

        Parameters
        ----------
        path : Sequence[str]
            The path part of the name. This is a sequence of
            strings, each indicating a deeper layer in the path hierarchy.
        transform_name : Optional[str]
            The name of the applied transformation. ``None`` means no
            transformation was applied.
        untransformed_name : str
            The plain part of the name.

        """
        self.path = tuple(path)
        self.untransformed_name = untransformed_name
        self.transform_name = transform_name

    @classmethod
    def from_name(cls, name: str) -> "NameParts":
        """Split a provided name into its parts and return them as ``NameParts``.

        Parameters
        ----------
        name : str
            The name that must be segmented into parts.

        Raises
        ------
        ValueError
            If the provided name is not valid.

        Returns
        -------
        NameParts
            The parts of the provided ``name`` are used to construct a
            ``NameParts`` instance which is returned.

        """
        split = name.split("/")
        path, original_name = split[:-1], split[-1]
        match = cls.NAME_RE.match(original_name)
        if not cls.is_valid_name(name):
            raise ValueError(cls.NAME_ERROR_MESSAGE.format(name))
        return cls(path, match["transform"], match["name"])  # type: ignore

    @property
    def original_name(self) -> str:
        """Return the name of the distribution without its preceeding path.

        Returns
        -------
        str
            The original name. This will include the transform and the
            untransformed parts of the name.
        """
        if self.is_transformed:
            return "__{}_{}".format(self.transform_name, self.untransformed_name)
        else:
            return self.untransformed_name

    @property
    def full_original_name(self) -> str:
        """Return the full name of the distribution with all three parts.

        Returns
        -------
        str
            The full name. This will include the path, transform and the
            untransformed parts of the name.
        """
        return "/".join(self.path + (self.original_name,))

    @property
    def full_untransformed_name(self) -> str:
        """Return the name of the distribution without its transform part.

        Returns
        -------
        str
            The path and the untransformed_name joined by a slash.
        """
        return "/".join(self.path + (self.untransformed_name,))

    @property
    def is_transformed(self) -> bool:
        """Return ``True`` if the ``transform`` part of the name is not ``None``."""
        return self.transform_name is not None

    def __repr__(self) -> str:
        """Return the ``NameParts`` ``full_original_name`` string representation."""
        return "<NameParts of {}>".format(self.full_original_name)

    def replace_transform(self, transform_name):
        """Replace the transform part of the name and return a new NameParts instance."""
        return self.__class__(self.path, transform_name, self.untransformed_name)


def get_data(filename):
    """Return a BytesIO object for a package data file.

    Parameters
    ----------
    filename : str
        file to load
    Returns
    -------
    BytesIO of the data
    """
    data_pkg = "notebooks"
    return io.BytesIO(pkgutil.get_data(data_pkg, os.path.join("data", filename)))
