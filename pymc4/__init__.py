from ._model import *
from ._random_variables import *

import tensorflow as _tf
from ._template_contexts import get_context as _get_context
from ._random_variables import RandomVariable as _RandomVariable


def _convert_rv_to_backend(d, dtype=None, name=None, as_ref=False):
    if isinstance(d, _RandomVariable):
        return d.as_tensor()
    return d


_tf.register_tensor_conversion_function(
    _RandomVariable, conversion_func=_convert_rv_to_backend, priority=0
)
