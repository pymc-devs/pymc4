"""Utility functions/classes for Variational Inference."""
import collections
import numpy as np
import tensorflow as tf
from typing import Dict

VarMap = collections.namedtuple("VarMap", "var, slc, shp, dtyp")


class ArrayOrdering:
    """
    An ordering for an array space.

    Parameters
    ----------
    free_rvs : dict
        Free random variables of the model
    """

    def __init__(self, free_rvs):
        self.free_rvs = free_rvs
        self.by_name = {}
        self.size = 0

        for name, tensor in free_rvs.items():
            flat_shape = int(np.prod(tensor.shape.as_list()))
            slc = slice(self.size, self.size + flat_shape)
            self.by_name[name] = VarMap(name, slc, tensor.shape, tensor.dtype)
            self.size += flat_shape

    def flatten(self) -> tf.Tensor:
        """Flattened view of parameters."""
        flattened_tensor = [tf.reshape(var, shape=[-1]) for var in self.free_rvs.values()]
        return tf.concat(flattened_tensor, axis=0)

    def split(self, flatten_tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Split view of parameters used to calculate log probability."""
        split_view = dict()
        for param in self.free_rvs:
            _, slc, shape, dtype = self.by_name[param]
            split_view[param] = tf.cast(tf.reshape(flatten_tensor[slc], shape), dtype)
        return split_view

    def split_samples(self, samples: tf.Tensor, n: int):
        """Split view of samples after drawing samples from posterior."""
        q_samples = dict()
        for param in self.free_rvs.keys():
            _, slc, shp, dtype = self.by_name[param]
            q_samples[param] = tf.cast(
                tf.reshape(samples[..., slc], tf.TensorShape([n] + shp.as_list())), dtype,
            )
        return q_samples
