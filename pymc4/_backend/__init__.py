import os
from pymc4._backend.base import Backend


__all__ = ["ops", "SUPPORTED_BACKEND_NAMES", "TENSORFLOW"]

ops: Backend  # to make type checkers a bit more happy
SUPPORTED_BACKEND_NAMES = {"tensorflow"}
TENSORFLOW = False

BACKEND = "tensorflow"
if "PYMC4_BACKEND" in os.environ:
    BACKEND = os.environ["PYMC4_BACKEND"]

if BACKEND == "tensorflow":
    from .tensorflow import TensorflowBackend

    ops = TensorflowBackend()
    TENSORFLOW = True
else:
    raise ImportError(
        "{} backend is not supported, please provide the supported one of {} "
        "in PYMC4_BACKEND env variable.".format(BACKEND, SUPPORTED_BACKEND_NAMES)
    )
