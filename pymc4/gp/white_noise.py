# TODO: Add White Noise Kernel!

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow.compat.v2 as tf  # pylint: disable=import-error

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import (
    PositiveSemidefiniteKernel,
)


class _WhiteNoise(PositiveSemidefiniteKernel):
    def __init__(self, noise, feature_ndims=1, validate_args=False, name="WhiteNoise"):
        parameters = dict(locals())
        with tf.name_scope(name):
            dtype = util.maybe_get_common_dtype([noise])
            self._noise = tensor_util.convert_nonref_to_tensor(noise, dtype=dtype, name="coef")
        super(_WhiteNoise, self).__init__(
            feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    def _apply(self, x1, x2, example_ndims=0):
        raise NotImplementedError("WhiteNoise kernel cannot be evaluated at a point!")

    def _matrix(self, x1, x2):
        shape = tf.broadcast_dynamic_shape(
            x1.shape[: -(1 + self.feature_ndims)], x2.shape[: -(1 + self.feature_ndims)],
        )
        expected = tf.linalg.eye(
            x1.shape[-(1 + self.feature_ndims)],
            x2.shape[-(1 + self.feature_ndims)],
            batch_shape=shape,
            dtype=self._dtype,
        )
        if self.noise is not None:
            noise = tf.convert_to_tensor(self._noise)
            noise = util.pad_shape_with_ones(noise, 2)
            expected = noise * expected
        return expected

    @property
    def noise(self):
        return self._noise

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape if self.noise is None else self.noise.shape

    def _batch_shape_tensor(self):
        return tf.TensorShape([]) if self.noise is None else self.noise.shape

    def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
        return super()._tensor(x1, x2, x1_example_ndims, x2_example_ndims)

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        for arg_name, arg in dict(noise=self.noise).items():
            if arg is not None and is_init != tensor_util.is_ref(arg):
                assertions.append(
                    assert_util.assert_positive(
                        arg, message="{} must be positive.".format(arg_name)
                    )
                )
        return assertions
