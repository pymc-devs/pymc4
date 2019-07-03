import functools
import re


def map_nested(fn, structure, cond=lambda obj: True):
    r"""
    Structure preserving nested map.

    Apply fn to an object in a possibly nested data structure and returns
    same structure with every element changed if condition satisfied.
    """

    def inner_map(obj):
        if cond(obj):
            return fn(obj)
        if isinstance(obj, (tuple, list)) and len(obj) > 0:
            return type(obj)(map(inner_map, obj))
        if isinstance(obj, dict) and len(obj) > 0:
            return dict(map(inner_map, obj.items()))
        return obj

    # After map_nested is called, a inner_map cell will exist. This cell
    # has a reference to the actual function inner_map, which has references
    # to a closure that has a reference to the inner_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return inner_map(structure)
    finally:
        inner_map = None


def merge_dicts(*dicts: dict, **kwargs: dict):
    """
    Merge dicts and assert their keys do not overlap.

    Parameters
    ----------
    dicts : dict
        Arbitrary number of dicts
    kwargs : dict
        Dict with keyword args for

    Returns
    -------
    dict
        Merged dict
    """
    for mappable in dicts:
        if set(mappable) & set(kwargs):
            raise ValueError(
                "Found duplicate keys in merge: {}".format(set(mappable) & set(kwargs))
            )
        kwargs.update(mappable)
    return kwargs


def biwrap(wrapper):
    @functools.wraps(wrapper)
    def enhanced(*args, **kwargs):
        is_bound_method = hasattr(args[0], wrapper.__name__) if args else False
        if is_bound_method:
            count = 1
        else:
            count = 0
        if len(args) > count:
            newfn = wrapper(*args, **kwargs)
            return newfn
        else:
            newwrapper = functools.partial(wrapper, *args, **kwargs)
            return newwrapper

    return enhanced


class NameParts(object):
    NAME_RE = re.compile(r"^(?:__(?P<transform>[^_]+)_)?(?P<name>[^_].*)$")
    NAME_ERROR_MESSAGE = (
        "Invalid name: `{}`, the correct one should look like: `__transform_name` or `name`, "
        "note only one underscore between the transform and actual name"
    )
    UNTRANSFORMED_NAME_ERROR_MESSAGE = (
        "Invalid name: `{}`, the correct one should look like: " "`name` without leading underscore"
    )
    __slots__ = ("path", "original_name", "untransformed_name", "transform_name")

    @classmethod
    def is_valid_untransformed_name(cls, name):
        match = cls.NAME_RE.match(name)
        return match is not None and match["transform"] is None

    @classmethod
    def is_valid_name(cls, name):
        match = cls.NAME_RE.match(name)
        return match is not None

    def __init__(self, name):
        split = name.split("/")
        path, original_name = split[:-1], split[-1]
        match = self.NAME_RE.match(original_name)
        if not self.is_valid_name(name):
            raise ValueError(self.NAME_ERROR_MESSAGE)
        self.path = tuple(path)
        self.original_name = original_name
        self.untransformed_name = match["name"]
        self.transform_name = match["transform"]

    @property
    def full_original_name(self):
        return "/".join(self.path + (self.original_name,))

    @property
    def full_untransformed_name(self):
        return "/".join(self.path + (self.untransformed_name,))

    @property
    def is_transformed(self):
        return self.transform_name is not None

    def __repr__(self):
        return "<NameParts of {}>".format(self.full_original_name)
