"""Constant Kernel and WhiteNoise Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

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


class _WhiteNoise(PositiveSemidefiniteKernel):
    def __init__(self, noise=None, feature_ndims=1, validate_args=False, name="WhiteNoise"):
        parameters = dict(locals())
        with tf.name_scope(name):
            dtype = util.maybe_get_common_dtype([noise])
            self._noise = tensor_util.convert_nonref_to_tensor(noise, dtype=dtype, name="noise")
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
            expected *= noise
        return expected

    @property
    def noise(self):
        return self._noise

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return scalar_shape if self.noise is None else self.noise.shape

    def _batch_shape_tensor(self):
        return [] if self.noise is None else tf.shape(self.noise)

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


class _Exponential(PositiveSemidefiniteKernel):
    def __init__(
        self,
        amplitude=None,
        length_scale=None,
        feature_ndims=1,
        validate_args=False,
        name="Exponential",
    ):
        parameters = dict(locals())
        with tf.name_scope(name):
            dtype = util.maybe_get_common_dtype([amplitude, length_scale])
            self._amplitude = tensor_util.convert_nonref_to_tensor(amplitude, dtype=dtype)
            self._length_scale = tensor_util.convert_nonref_to_tensor(length_scale, dtype=dtype)
        super(_Exponential, self).__init__(
            feature_ndims=feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    @property
    def length_scale(self):
        return self._length_scale

    @property
    def amplitude(self):
        return self._amplitude

    def _apply(self, x1, x2, example_ndims=0):
        sqdist = util.sum_rightmost_ndims_preserving_shape(
            tf.math.squared_difference(x1, x2), self.feature_ndims
        )
        ndist = -0.5 * tf.sqrt(sqdist + 1e-12)
        if self.length_scale is not None:
            length_scale = tf.convert_to_tensor(self._length_scale)
            length_scale = util.pad_shape_with_ones(length_scale, example_ndims)
            ndist /= length_scale ** 2

        if self.amplitude is not None:
            amplitude = tf.convert_to_tensor(self._amplitude)
            amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
            return amplitude ** 2 * tf.exp(ndist)

        return tf.exp(ndist)

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            scalar_shape if self.amplitude is None else self.amplitude.shape,
            scalar_shape if self.length_scale is None else self.length_scale.shape,
        )

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            [] if self.amplitude is None else tf.shape(self.amplitude),
            [] if self.length_scale is None else tf.shape(self.length_scale),
        )

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        for arg_name, arg in dict(amplitude=self.amplitude, length_scale=self.length_scale).items():
            if arg is not None and is_init != tensor_util.is_ref(arg):
                assertions.append(
                    assert_util.assert_positive(
                        arg, message="{} must be positive.".format(arg_name)
                    )
                )
        return assertions


class _Gibbs(PositiveSemidefiniteKernel):
    def __init__(
        self,
        length_scale_fn=None,
        fn_args=None,
        feature_ndims=1,
        dtype=tf.float32,
        validate_args=False,
        name="Gibbs",
    ):
        parameters = locals()
        with tf.name_scope(name):
            self._length_scale_fn = length_scale_fn
            self._fn_args = fn_args
        super(_Gibbs, self).__init__(
            feature_ndims=feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    def _log_apply(self, lx1, lx2):
        loglx1 = tf.math.log(lx1)
        loglx2 = tf.math.log(lx2)
        lognum = util.sum_rightmost_ndims_preserving_shape(
            loglx1 + loglx2 + math.log(2.0), self.feature_ndims
        )
        logdenom = util.sum_rightmost_ndims_preserving_shape(
            tf.math.log(lx1 ** 2 + lx2 ** 2), self.feature_ndims
        )
        return tf.exp(0.5 * (lognum - logdenom))

    def _fast_apply(self, x1, x2):
        lx1 = tf.convert_to_tensor(self._length_scale_fn(x1, *self._fn_args))
        lx2 = tf.convert_to_tensor(self._length_scale_fn(x2, *self._fn_args))
        lx12, lx22 = lx1 ** 2, lx2 ** 2
        scal = util.sum_rightmost_ndims_preserving_shape(
            tf.sqrt(2 * lx1 * lx2 / (lx12 + lx22)), self.feature_ndims
        )
        sqdist = tf.math.squared_difference(x1, x2)
        sqdist /= lx12 + lx22
        sqdist = util.sum_rightmost_ndims_preserving_shape(sqdist, self.feature_ndims)
        return scal * tf.exp(-sqdist)

    def _apply(self, x1, x2, example_ndims=0):
        if self._length_scale_fn is not None:
            if x1.shape[-1] == 1 and self.feature_ndims == 1:
                return self._fast_apply(x1, x2)
            lx1 = tf.convert_to_tensor(self._length_scale_fn(x1, *self._fn_args))
            lx2 = tf.convert_to_tensor(self._length_scale_fn(x2, *self._fn_args))
            scal = self._log_apply(lx1, lx2)
            sqdist = tf.math.squared_difference(x1, x2)
            sqdist /= lx1 ** 2 + lx2 ** 2
            sqdist = util.sum_rightmost_ndims_preserving_shape(sqdist, self.feature_ndims)
            return scal * tf.exp(-sqdist)
        sqdist = util.sum_rightmost_ndims_preserving_shape(
            tf.math.squared_difference(x1, x2), self.feature_ndims
        )
        return tf.exp(-sqdist / 2)

    def _batch_shape(self):
        return tf.TensorShape([])

    def _batch_shape_tensor(self):
        return tf.shape([])

    def _parameter_control_dependencies(self, is_init):
        return []


class _Cosine(PositiveSemidefiniteKernel):
    def __init__(
        self,
        length_scale=None,
        amplitude=None,
        feature_ndims=1,
        validate_args=False,
        name="Cosine",
    ):
        parameters = locals()
        with tf.name_scope(name):
            dtype = util.maybe_get_common_dtype([length_scale, amplitude])
            self._length_scale = tensor_util.convert_nonref_to_tensor(length_scale, dtype=dtype)
            self._amplitude = tensor_util.convert_nonref_to_tensor(amplitude, dtype=dtype)
        super(_Cosine, self).__init__(
            feature_ndims=feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    @property
    def length_scale(self):
        return self._length_scale

    @property
    def amplitude(self):
        return self._amplitude

    def _apply(self, x1, x2, example_ndims=0):
        component = (
            2.0
            * math.pi
            * tf.sqrt(
                util.sum_rightmost_ndims_preserving_shape(
                    tf.math.squared_difference(x1, x2), self.feature_ndims
                )
            )
        )
        if self.length_scale is not None:
            length_scale = tf.convert_to_tensor(self._length_scale)
            length_scale = util.pad_shape_with_ones(length_scale, example_ndims)
            component /= length_scale ** 2
        if self.amplitude is not None:
            amplitude = tf.convert_to_tensor(self._amplitude)
            amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
            return amplitude ** 2 * tf.math.cos(component)
        return tf.math.cos(component)

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            scalar_shape if self._amplitude is None else self._amplitude.shape,
            tf.broadcast_static_shape(
                scalar_shape if self._length_scale is None else self._length_scale.shape,
                scalar_shape if self._period is None else self._period.shape,
            ),
        )

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            tf.broadcast_dynamic_shape(
                [] if self.amplitude is None else tf.shape(self.amplitude),
                [] if self.length_scale is None else tf.shape(self.length_scale),
            ),
            [] if self.period is None else tf.shape(self.period),
        )

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        for arg_name, arg in dict(
            amplitude=self.amplitude, length_scale=self.length_scale, period=self.period
        ).items():
            if arg is not None and is_init != tensor_util.is_ref(arg):
                assertions.append(
                    assert_util.assert_positive(
                        arg, message="{} must be positive.".format(arg_name)
                    )
                )
        return assertions


# FIXME: This kernel is not implemented currently as tensorflow doesn't allow
#        slicing with tensors or arrays. Any help would be appriciated.
# class Coregion(PositiveSemidefiniteKernel):
#     def __init__(
#         self, W=None, kappa=None, B=None, feature_ndims=None, validate_args=False, name="Coregion"
#     ):
#         parameters = locals()
#         with tf.name_scope(name):
#             dtype = util.maybe_get_common_dtype([W, kappa, B])
#             self._W = tensor_util.convert_nonref_to_tensor(W)
#             self._kappa = tensor_util.convert_nonref_to_tensor(kappa)
#             if B is not None:
#                 self._B = tensor_util.convert_nonref_to_tensor(B)
#             else:
#                 self._B = tf.linalg.matmul(self._W, self._W, transpose_b=True) + tf.linalg.diag(
#                     self._kappa
#                 )
#         super().__init__(
#             feature_ndims=feature_ndims,
#             dtype=dtype,
#             name=name,
#             validate_args=validate_args,
#             parameters=parameters,
#         )

#     @property
#     def W(self):
#         return self._W

#     @property
#     def B(self):
#         return self._B

#     @property
#     def kappa(self):
#         return self._kappa

#     def _apply(self, x1, x2, example_ndims=0):
#         raise NotImplementedError("Coregion doesn't have a point evaluation scheme")

#     def _matrix(self, x1, x2):
#         x1_idx = tf.cast(x1, tf.int32)
#         x2_idx = tf.cast(x2, tf.int32).T
#         return tf.gather_nd(self._B,)


class _ScaledCov(PositiveSemidefiniteKernel):
    def __init__(
        self,
        kernel=None,
        scaling_fn=None,
        fn_args=None,
        feature_ndims=1,
        validate_args=False,
        name="ScaledCov",
    ):
        parameters = locals()
        with tf.name_scope(name):
            self._kernel = kernel
            self._scaling_fn = scaling_fn
            if fn_args is None:
                fn_args = tuple()
            self._fn_args = fn_args
        super(_ScaledCov, self).__init__(
            feature_ndims=feature_ndims,
            dtype=kernel._dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    @property
    def kernel(self):
        return self._kernel

    @property
    def scaling_fn(self):
        return self._scaling_fn

    @property
    def fn_Args(self):
        return self._fn_args

    def _apply(self, x1, x2, example_ndims=0):
        cov = self._kernel._apply(x1, x2, example_ndims)
        if self._scaling_fn is not None:
            scal_x1 = tf.convert_to_tensor(self._scaling_fn(x1, *self._fn_args))
            scal_x2 = tf.convert_to_tensor(self._scaling_fn(x2, *self._fn_args))
            scal = util.sum_rightmost_ndims_preserving_shape(scal_x1 * scal_x2, self._feature_ndims)
            return scal * cov
        return cov

    def _matrix(self, x1, x2):
        cov = self._kernel._matrix(x1, x2)
        if self._scaling_fn is not None:
            scal_x1 = util.pad_shape_with_ones(
                tf.convert_to_tensor(self._scaling_fn(x1, *self._fn_args)),
                ndims=1,
                start=-(self._feature_ndims + 1),
            )
            scal_x2 = util.pad_shape_with_ones(
                tf.convert_to_tensor(self._scaling_fn(x2, *self._fn_args)),
                ndims=1,
                start=-(self._feature_ndims + 2),
            )
            scal = util.sum_rightmost_ndims_preserving_shape(
                scal_x1 * scal_x2, ndims=self._feature_ndims
            )
            return scal * cov
        return cov

    def _batch_shape(self):
        return self._kernel.batch_shape

    def _batch_shape_tensor(self):
        return self._kernel._batch_shape_tensor()

    def _parameter_control_dependencies(self, is_init):
        return self._kernel._parameter_control_dependencies(is_init=is_init)
