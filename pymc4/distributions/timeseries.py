import tensorflow as tf
from tensorflow_probability import sts
from pymc4.distributions.distribution import ContinuousDistribution


class AR(ContinuousDistribution):
    r"""Autoregressive process with `order` lags.

    Parameters
    ----------
    num_timesteps : int Tensor
            Total number of timesteps to model.
    coefficients : float Tensor
            Autoregressive coefficients of shape `concat(batch_shape, [order])`.
    level_scale : Scalar float Tensor
            Standard deviation of the transition noise at each step
            (any additional dimensions are treated as batch
            dimensions).
    initial_step : (Optional) int Tensor
            Starting timestep (Default value: 0).

    Examples
    --------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.AR('x', num_timesteps=50, coefficients=[0.2, -0.8], level_scale=-0.2)
    """

    def __init__(self, name, num_timesteps, coefficients, level_scale, initial_step=0, **kwargs):
        super().__init__(name, num_timesteps=num_timesteps, coefficients=coefficients,
                         level_scale=level_scale, initial_step=initial_step, **kwargs)

    @staticmethod
    def _init_distribution(conditions: dict):
        num_timesteps = conditions["num_timesteps"]
        coefficients = conditions["coefficients"]
        level_scale = conditions["level_scale"]
        initial_step = conditions["initial_step"]
        coefficients = tf.convert_to_tensor(value=coefficients, name='coefficients')
        order = tf.compat.dimension_value(coefficients.shape[-1])
        time_series_object = sts.Autoregressive(order=order)
        distribution = time_series_object.make_state_space_model(
            num_timesteps=num_timesteps,
            param_vals={'coefficients': coefficients,
                        'level_scale': level_scale},
            initial_state_prior=time_series_object.initial_state_prior,
            initial_step=initial_step)
        return distribution
