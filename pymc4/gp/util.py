import tensorflow as tf


def stabilize(K):
    """Add a diagonal shift to a covarience matrix"""
    return tf.linalg.set_diag(K, tf.linalg.diag_part(K) + 1e-6)
