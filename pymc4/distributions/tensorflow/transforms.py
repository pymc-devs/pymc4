from pymc4.distributions import abstract
from tensorflow_probability import bijectors as tfb

__all__ = ["Log"]


class Log(abstract.transforms.Log):
    def __init__(self):
        # NOTE: We actually need the inverse to match PyMC3, do we?
        self._backend_transform = tfb.Exp()

    def forward(self, x):
        return self._backend_transform.inverse(x)

    def inverse(self, z):
        return self._backend_transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._backend_transform.inverse_log_det_jacobian(
            x, self._backend_transform.inverse_min_event_ndims
        )

    def inverse_log_det_jacobian(self, z):
        return self._backend_transform.forward_log_det_jacobian(
            z, self._backend_transform.forward_min_event_ndims
        )
