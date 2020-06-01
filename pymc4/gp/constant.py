"""Constant kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf  # pylint: disable=import-error

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import (
    PositiveSemidefiniteKernel,
)


class _Constant(PositiveSemidefiniteKernel):
    def __init__(self, coef=None, feature_ndims=1, validate_args=False, name="Constant"):
        parameters = dict(locals())
        with tf.name_scope(name):
            dtype = util.maybe_get_common_dtype([coef])
            self._coef = tensor_util.convert_nonref_to_tensor(coef, dtype=dtype, name="coef")
        super(_Constant, self).__init__(
            feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    def _apply(self, x1, x2, example_ndims=0):
        shape = tf.broadcast_dynamic_shape(
            x1.shape[: -(self.feature_ndims)], x2.shape[: -(self.feature_ndims)],
        )
        expected = tf.ones(shape, dtype=self._dtype)
        if self.coef is not None:
            coef = tf.convert_to_tensor(self._coef)
            coef = util.pad_shape_with_ones(coef, example_ndims)
            expected *= coef
        return expected

    @property
    def coef(self):
        return self._coef

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape if self.coef is None else self.coef.shape

    def _batch_shape_tensor(self):
        return tf.TensorShape([]) if self.coef is None else self.coef.shape

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        for arg_name, arg in dict(coef=self.coef).items():
            if arg is not None and is_init != tensor_util.is_ref(arg):
                assertions.append(
                    assert_util.assert_positive(
                        arg, message="{} must be positive.".format(arg_name)
                    )
                )
        return assertions
