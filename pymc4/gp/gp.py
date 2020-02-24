from abc import abstractmethod
import tensorflow as tf
# import tensorflow_probability as tfp
# import numpy as np
# from .cov import ExpQuad
# from .mean import Zero
from ..distributions import MvNormal
from .util import stabilize

class BaseGP:
    def __init__(self, mean_fn, cov_fn):
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn
    
    @abstractmethod
    def prior(self, name, X, **kwargs):
        raise NotImplementedError("Your GP should override this method")

    @abstractmethod
    def conditional(self, name, Xnew, **kwargs):
        raise NotImplementedError("Your GP should override this method")

    def predict(self, name, Xnew, **kwargs):
        raise NotImplementedError("Your GP should override this method")

    def marginal_likelihood(self, name, X, **kwargs):
        raise NotImplementedError("Your GP should override this method")


class LatentGP(BaseGP):
    def __init__(self, mean_fn, cov_fn):
        super().__init__(mean_fn=mean_fn, cov_fn=cov_fn)

    def _build_prior(self, name, X, **kwargs):
        mu = self.mean_fn(X)
        cov = stabilize(self.cov_fn(X, X))
        # TODO: don't forget to handle for multiple feature_ndims
        # shape = kwargs.pop("shape", X.shape[:-1])
        f = MvNormal(name, loc=mu, covariance_matrix=cov, **kwargs)
        return f

    def _get_given_vals(self, given):
        if given is None:
            given = {}
        if 'gp' in given:
            cov_total = given['gp'].cov_fn
            mean_total = given['gp'].mean_fn
        else:
            cov_total = self.cov_fn
            mean_total = self.mean_fn
        if all(val in given for val in ['X', 'f']):
            X, f = given['X'], given['f']
        else:
            X, f = self.X, self.f
        return X, f, cov_total, mean_total

    def _build_conditional(self, Xnew, X, f, cov_total, mean_total):
        Kxx = stabilize(cov_total(X, X))
        Kxs = stabilize(self.cov_fn(X, Xnew))
        L = tf.linalg.cholesky(Kxx)
        A = tf.linalg.cholesky_solve(L, Kxs)
        # NOTE and TODO: We need to somehow convert ``f`` PyMC4 distribution to tensor
        v = tf.linalg.cholesky_solve(L, f - mean_total(X))
        mu = self.mean_fn(Xnew) + A.T @ v
        Kss = self.cov_fn(Xnew, Xnew)
        cov = Kss - A.T @ A
        return mu, cov

    def prior(self, name, X, **kwargs):
        f = self._build_prior(name, X, **kwargs)
        self.X = X
        self.f = f
        return f

    def conditional(self, name, Xnew, given=None, **kwargs):
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, *givens)
        # TODO: don't forget to handle for multiple feature_ndims
        shape = kwargs.pop("shape", Xnew.shape[:-1])
        return MvNormal(name, mu, cov, shape=shape, **kwargs)
