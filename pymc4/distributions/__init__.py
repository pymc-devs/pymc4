from pymc4 import _backend
from pymc4.distributions import abstract

if _backend.TENSORFLOW:
    from .tensorflow import *  # pylint: disable=wildcard-import
else:
    raise ImportError("Backend {} not supported".format(_backend.BACKEND))
