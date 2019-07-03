import tensorflow as tf
from pymc4.distributions import abstract


class Log(abstract.transforms.Log):
    # do not use tfp bijectors if transform is not too complicated
    def forward(self, x):
        return tf.math.log(x)

    def backward(self, z):
        return tf.math.exp(z)

    def jacobian_log_det(self, x):
        return -tf.math.log(x)

    def inverse_jacobian_log_det(self, z):
        return z
