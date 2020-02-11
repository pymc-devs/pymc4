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
            order = coefficients.shape[-1] (order>0)
    level_scale : Scalar float Tensor
            Standard deviation of the transition noise at each step
            (any additional dimensions are treated as batch
            dimensions).
    initial_state : (Optional) float Tensor
            Corresponding values of size `order` for
            imagined timesteps before the initial step.
    observed_time_series : (Optional) float Tensor
            Observed time series
    initial_step : (Optional) int Tensor
            Starting timestep (Default value: 0).

    Examples
    --------
    .. code-block:: python
        @pm.model
        def model():
            x = pm.AR('x', num_timesteps=50, coefficients=[0.2, -0.8], level_scale=-0.2)
    """

    def __init__(
        self,
        name,
        num_timesteps,
        coefficients,
        level_scale,
        initial_state=None,
        observed_time_series=None,
        initial_step=0,
        **kwargs,
    ):
        if initial_state is not None and observed_time_series is None:
            raise TypeError(
                "Missing parameter 'observed_time_series' if 'initial_state' is specified"
            )
        super().__init__(
            name,
            num_timesteps=num_timesteps,
            coefficients=coefficients,
            level_scale=level_scale,
            initial_state=initial_state,
            observed_time_series=observed_time_series,
            initial_step=initial_step,
            **kwargs,
        )

    @staticmethod
    def _init_distribution(conditions: dict):
        num_timesteps = conditions["num_timesteps"]
        coefficients = conditions["coefficients"]
        level_scale = conditions["level_scale"]
        initial_state = conditions["initial_state"]
        observed_time_series = conditions["observed_time_series"]
        initial_step = conditions["initial_step"]

        coefficients = tf.convert_to_tensor(value=coefficients, name="coefficients")
        order = tf.compat.dimension_value(coefficients.shape[-1])

        if initial_step is not None and observed_time_series is not None:
            initial_state = tf.convert_to_tensor(value=initial_state)
            observed_time_series = tf.convert_to_tensor(value=observed_time_series)
            observed_time_series = tf.concat([initial_state, observed_time_series], axis=0)

        time_series_object = sts.Autoregressive(
            order=order, observed_time_series=observed_time_series
        )
        distribution = time_series_object.make_state_space_model(
            num_timesteps=num_timesteps,
            param_vals={"coefficients": coefficients, "level_scale": level_scale},
            initial_state_prior=time_series_object.initial_state_prior,
            initial_step=initial_step,
        )
        return distribution
