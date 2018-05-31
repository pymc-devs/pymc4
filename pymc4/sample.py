import tensorflow as tf
from pymc4 import Model
import numpy as np
import xarray as xr
from pymc4 import Model
import tqdm

__all__ = ["sample"]


def sample(draws=1000, tune=500, as_xarray=True):
    """To sample from a defined model with one or more random variable.
    """
    model = Model.get_context()
    array = []
    with tf.Session() as sess:
        for i in tqdm.trange(draws+tune):

            # Sampling methods are applied here.
            # Directly using tensorflow's default sampling method for now
            array.append([i.eval() for i in model.named_vars.values()])
    if as_xarray:
        return (xr.DataArray(
                            data=array[tune:],
                            dims=("Val", "RV"),
                            coords={"RV": list(model.named_vars.keys())}
                            )
                )
    else:
        return np.array(array[Tune])
