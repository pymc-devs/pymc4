import tensorflow as tf
from .. import abstract


class Exp(abstract.transforms.Exp):
    # do not use tfp bijectors if transform is not too complicated
    def forward(self, x):
        return tf.math.exp(x)

    def backward(self, z):
        return tf.math.log(z)

    def jacobian_log_det(self, x):
        return x

    def inverse_jacobian_log_det(self, z):
        return -tf.math.log(z)
