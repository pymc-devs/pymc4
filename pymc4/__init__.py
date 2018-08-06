__version__ = "0.0.1"

from tensorflow_probability.python.distributions import *  # pylint: disable=wildcard-import

from . import model
from .model import (
    Model,
    inline
)
from . import inference
from .inference import (
    sampling
)
from .inference.sampling.sample import sample
from . import util

from .util.generated_random_variable import *  # pylint: disable=wildcard-import
