"""
Gaussian Process Models present in PyMC4's Gaussian Process module.

"""


import tensorflow as tf

# import tensorflow_probability as tfp
# import numpy as np
# from .cov import ExpQuad
# from .mean import Zero
from ..distributions import MvNormal, Normal
from .util import stabilize


__all__ = ["LatentGP"]


class BaseGP:
    def __init__(self, mean_fn, cov_fn):
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def prior(self, name, X, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, **kwargs):
        raise NotImplementedError

    def predict(self, name, Xnew, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, **kwargs):
        raise NotImplementedError


class LatentGP(BaseGP):
    def __init__(self, mean_fn, cov_fn):
        super().__init__(mean_fn=mean_fn, cov_fn=cov_fn)

    def _is_univariate(self, X):
        return X.shape[-(self.cov_fn.feature_ndims + 1)] == 1

    def _build_prior(self, name, X, **kwargs):
        mu = self.mean_fn(X)
        cov = stabilize(self.cov_fn(X, X))
        if self._is_univariate(X):
            return Normal(loc=tf.squeeze(mu), scale=tf.squeeze(cov, axis=[-1, -2]), **kwargs)
        return MvNormal(name, loc=mu, covariance_matrix=cov, **kwargs)

    def _get_given_vals(self, given):
        # TODO: make this work if given=None
        if given is None:
            given = {}
        if "gp" in given:
            cov_total = given["gp"].cov_fn
            mean_total = given["gp"].mean_fn
        else:
            cov_total = self.cov_fn
            mean_total = self.mean_fn
        if all(val in given for val in ["X", "f"]):
            X, f = given["X"], given["f"]
        else:
            X, f = self.X, self.f
        return X, f, cov_total, mean_total

    def _build_conditional(self, Xnew, X, f, cov_total, mean_total):
        # TODO: error handling
        if not tf.is_tensor(f):
            raise ValueError("Prior `f` must be a numpy array or tensor.")
        Kxx = cov_total(X, X)
        Kxs = self.cov_fn(X, Xnew)
        L = tf.linalg.cholesky(stabilize(Kxx))
        A = tf.linalg.solve(L, Kxs)
        # We add a `newaxis` to make the shape of mean_total(X)
        # [batch_shape, event_shape, 1] which is consistent with
        # the shape `tf.linalg.solve` accepts.
        v = tf.linalg.solve(L, (f - mean_total(X))[..., tf.newaxis])
        # Add a axis to avoid broadcasting
        mu = self.mean_fn(Xnew)[..., tf.newaxis] + tf.linalg.matrix_transpose(A) @ v
        Kss = self.cov_fn(Xnew, Xnew)
        cov = Kss - tf.linalg.matrix_transpose(A) @ A
        # Return the stabilized covarience matrix and squeeze the 
        # last dimention of the mean off.
        return tf.squeeze(mu, axis=[-1]), stabilize(cov)

    def prior(self, name, X, **kwargs):
        """Evaluate prior distribution at given data points."""
        f = self._build_prior(name, X, **kwargs)
        self.X = X
        self.f = f
        return f

    def conditional(self, name, Xnew, given, **kwargs):
        """Evaluate conditional distribution at new data points."""
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, *givens)
        if self._is_univariate(Xnew):
            return Normal(name=name, loc=tf.squeeze(mu, [-1]), cov=tf.squeeze(cov, [-1, -2]))
        return MvNormal(name=name, loc=mu, covariance_matrix=cov, **kwargs)
