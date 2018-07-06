import collections
import numpy as np
import tensorflow as tf


def make_shared_vectorized_input(rvs, test_values):
    dim = 0
    for shape in rvs.items():
        dim += np.prod(shape.as_list())
    init = np.empty(dim)
    j = 0
    for name, shape in rvs.items():
        d = np.prod(shape.as_list())
        init[j:d] = np.asarray(test_values[name]).flatten()
        j += d
    vec_shape = (dim, )
    with tf.variable_scope('vectorize'):
        vec = tf.get_variable('vec_state', vec_shape, initializer=tf.constant_initializer(init))
        j = 0
        mapping = collections.OrderedDict()
        with tf.name_scope('slice'):
            for name, shape in rvs.items():
                d = np.prod(shape.as_list())
                mapping[name] = tf.reshape(vec[j:d], shape, name=name)
                j += d
    return vec, mapping
