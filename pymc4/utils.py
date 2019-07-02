import functools


def map_nested(fn, structure, cond=lambda obj: True):
    r"""
    Applies fn to an object in a possibly nested data structure and returns same
    structure with every element changed if condition satisfied.
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
    Helper function to merge dicts without overlapping keys

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
