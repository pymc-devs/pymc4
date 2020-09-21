import enum
from typing import Optional

from tensorflow_probability import bijectors as tfb


__all__ = ["Log", "Sigmoid", "LowerBound", "UpperBound", "Interval"]


class JacobianPreference(enum.Enum):
    Forward = "Forward"
    Backward = "Backward"


class Transform:
    """
    Baseclass to define a bijective transformation of a distribution.

    Parameters
    ----------
    transform : tfp.bijectors.bijector
        The bijector that is called for the transformation.
    event_ndims : int
        The number of event dimensions of the distribution that has to be transformed.
        This is normally automatically set during the initialization of the
        PyMC4 distribution
    """

    name: Optional[str] = None
    jacobian_preference = JacobianPreference.Forward

    def __init__(self, transform=None, event_ndims=None):
        self._transform = transform
        self.event_ndims = event_ndims

    @property
    def _untransformed_event_ndims(self):
        """
        The length of the event_shape of the untransformed distribution. If set to None,
        it can lead to errors in case of non autobatched sampling.
        """
        if self.event_ndims is None:
            return self._min_event_ndims
        else:
            return self.event_ndims

    @property
    def _transformed_event_ndims(self):
        """ The length of the event_shape of the transformed distribution"""
        if self.event_ndims is None:
            return self._transform.inverse_event_ndims(self._min_event_ndims)
        else:
            return self._transform.inverse_event_ndims(self.event_ndims)

    @property
    def _min_event_ndims(self):
        return NotImplementedError

    def forward(self, x):
        """
        Forward of a bijector.

        Applies transformation forward to input variable `x`.
        When transform is used on some distribution `p`, it will transform the random variable `x` after sampling
        from `p`.

        Parameters
        ----------
        x : tensor
            Input tensor to be transformed.

        Returns
        -------
        tensor
            Transformed tensor.
        """
        raise NotImplementedError

    def inverse(self, z):
        """
        Backward of a bijector.

        Applies inverse of transformation to input variable `z`.
        When transform is used on some distribution `p`, which has observed values `z`, it is used to
        transform the values of `z` correctly to the support of `p`.

        Parameters
        ----------
        z : tensor
            Input tensor to be inverse transformed.

        Returns
        -------
        tensor
            Inverse transformed tensor.
        """
        raise NotImplementedError

    def forward_log_det_jacobian(self, x):
        """
        Calculate logarithm of the absolute value of the Jacobian determinant for input `x`.

        Parameters
        ----------
        x : tensor
            Input to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `x` w.r.t. this transform.
        """
        raise NotImplementedError

    def inverse_log_det_jacobian(self, z):
        """
        Calculate logarithm of the absolute value of the Jacobian determinant for output `z`.

        Parameters
        ----------
        z : tensor
            Output to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `z` w.r.t. this transform.

        Notes
        -----
        May be desired to be implemented efficiently
        """
        raise -self.forward_log_det_jacobian(self.inverse(z))


class Invert(Transform):
    def __init__(self, transform, **kwargs):
        if transform.jacobian_preference == JacobianPreference.Forward:
            self.jacobian_preference = JacobianPreference.Backward
        else:
            self.jacobian_preference = JacobianPreference.Forward
        super().__init__(transform, **kwargs)

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(x, self._untransformed_event_ndims)

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(z, self._transformed_event_ndims)


class BackwardTransform(Transform):
    """
    Base class for Transforms with Jacobian Preference as Backward.
    Backward means that the transformed values are in the domain of the specified function
    and the untransformed values in the codomain.
    """

    JacobianPreference = JacobianPreference.Backward

    def __init__(self, transform, **kwargs):
        super().__init__(transform, **kwargs)

    @property
    def _min_event_ndims(self):
        return self._transform._inverse_min_event_ndims

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(x, self._untransformed_event_ndims)

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(z, self._transformed_event_ndims)


class Log(BackwardTransform):
    name = "log"

    def __init__(self, **kwargs):
        # NOTE: We actually need the inverse to match PyMC3, do we?
        transform = tfb.Exp()
        super().__init__(transform, **kwargs)


class Sigmoid(BackwardTransform):
    name = "sigmoid"

    def __init__(self, **kwargs):
        transform = tfb.Sigmoid()
        super().__init__(transform, **kwargs)


class LowerBound(BackwardTransform):
    """"Transformation to interval [lower_limit, inf]"""

    name = "lowerbound"

    def __init__(self, lower_limit, **kwargs):
        transform = tfb.Chain([tfb.Shift(lower_limit), tfb.Exp()])
        super().__init__(transform, **kwargs)


class UpperBound(BackwardTransform):
    """"Transformation to interval [-inf, upper_limit]"""

    name = "upperbound"

    def __init__(self, upper_limit, **kwargs):
        transform = tfb.Chain([tfb.Shift(upper_limit), tfb.Scale(-1), tfb.Exp()])
        super().__init__(transform, **kwargs)


class Interval(BackwardTransform):
    """"Transformation to interval [lower_limit, upper_limit]"""

    name = "interval"

    def __init__(self, lower_limit, upper_limit, **kwargs):
        transform = tfb.Sigmoid(low=lower_limit, high=upper_limit)
        super().__init__(transform, **kwargs)
