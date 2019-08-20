from pymc4 import _backend

if _backend.TENSORFLOW:
    from .tensorflow import *  # pylint: disable=wildcard-import
else:
    raise ImportError("Backend {} not supported".format(_backend.BACKEND))
