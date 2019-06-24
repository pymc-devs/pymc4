import os
from . import abstract

_backend = "tensorflow"
if "PYMC_BACKEND" in os.environ:
    _backend = os.environ["PYMC_BACKEND"]

if _backend == "tensorflow":
    from .tensorflow import *
else:
    raise ImportError("Backend {} not supported".format(_backend))
