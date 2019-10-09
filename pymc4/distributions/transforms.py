import enum
from typing import Optional

from tensorflow_probability import bijectors as tfb

__all__ = ["Log"]


class JacobianPreference(enum.Enum):
    Forward = "Forward"
    Backward = "Backward"


class Transform:
    name: Optional[str] = None
    jacobian_preference = JacobianPreference.Forward

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
    def __init__(self, transform):
        if transform.jacobian_preference == JacobianPreference.Forward:
            self.jacobian_preference = JacobianPreference.Backward
        else:
            self.jacobian_preference = JacobianPreference.Forward
        self.transform = transform

    def forward(self, x):
        return self.transform.inverse(x)

    def inverse(self, z):
        return self.transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self.transform.inverse_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, z):
        return self.transform.forward_log_det_jacobian(z)


class Log(Transform):
    name = "log"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self):
        # NOTE: We actually need the inverse to match PyMC3, do we?
        self._transform = tfb.Exp()

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(x, self._transform.inverse_min_event_ndims)

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(z, self._transform.forward_min_event_ndims)
