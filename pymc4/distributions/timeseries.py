import warnings
import tensorflow as tf
from tensorflow_probability import sts
from tensorflow_probability import distributions as tfd
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
    initial_step : (Optional) int Tensor
            Starting timestep (Default value: 0).

    Examples
    --------
    >>> import pymc4 as pm
    >>> @pm.model
    ... def model():
    ...     x = yield pm.AR('x', num_timesteps=50, coefficients=[0.2, -0.8], level_scale=-0.2)
    """

    def __init__(
        self,
        name,
        num_timesteps,
        coefficients,
        level_scale,
        initial_state=None,
        initial_step=0,
        **kwargs,
    ):
        super().__init__(
            name,
            num_timesteps=num_timesteps,
            coefficients=coefficients,
            level_scale=level_scale,
            initial_state=initial_state,
            initial_step=initial_step,
            **kwargs,
        )

    @classmethod
    def unpack_conditions(cls, **kwargs):
        conditions, base_parameters = super().unpack_conditions(**kwargs)
        warnings.warn(
            "At the moment, the Autoregressive distribution does not accept the initialization "
            "arguments: dtype, allow_nan_stats or validate_args. Any of those keyword arguments "
            "passed during initialization will be ignored."
        )
        return conditions, {}

    @staticmethod
    def _init_distribution(conditions: dict, **kwargs):
        num_timesteps = conditions["num_timesteps"]
        coefficients = conditions["coefficients"]
        level_scale = conditions["level_scale"]
        initial_state = conditions["initial_state"]
        initial_step = conditions["initial_step"]

        coefficients = tf.convert_to_tensor(value=coefficients, name="coefficients")
        order = tf.compat.dimension_value(coefficients.shape[-1])

        time_series_object = sts.Autoregressive(order=order)
        distribution = time_series_object.make_state_space_model(
            num_timesteps=num_timesteps,
            param_vals={"coefficients": coefficients, "level_scale": level_scale},
            initial_state_prior=tfd.MultivariateNormalDiag(
                loc=initial_state, scale_diag=[1e-6] * order
            ),
            initial_step=initial_step,
        )
        return distribution
