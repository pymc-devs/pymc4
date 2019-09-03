import tensorflow as tf
from pymc4.distributions import abstract
from tensorflow_probability import bijectors as tfb

__all__ = ["Log"]


class Log(abstract.transforms.Log):
    # do not use tfp bijectors if transform is not too complicated
    def forward(self, x):
        return tfb.Exp().forward(x)
        #return tf.math.log(x)

    def backward(self, z):
        return tfb.Exp().inverse(z)
        #return tf.math.exp(z)

    def jacobian_log_det(self, x):
        return tfb.Exp().forward_log_det_jacobian(x, 1)
        #return -tf.math.log(x)

    def inverse_jacobian_log_det(self, z):
        return z
        #return z
