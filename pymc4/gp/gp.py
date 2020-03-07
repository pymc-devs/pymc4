"""
Gaussian Process Models present in PyMC4's Gaussian Process module.

"""


import tensorflow as tf
from .cov import ExpQuad
from .mean import Zero
from ..distributions import MvNormal, Normal
from .util import stabilize


__all__ = ["LatentGP"]


class BaseGP:
    def __init__(self, mean_fn=Zero(1), cov_fn=ExpQuad(1.0, 1.0, 1)):
        if mean_fn.feature_ndims != cov_fn.feature_ndims:
            raise ValueError(
                "The feature_ndims of mean and covariance functions should be the same"
            )
        self.feature_ndims = mean_fn.feature_ndims
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
    r"""Latent Gaussian process.

    The `gp.LatentGP` class is a direct implementation of a GP.  No additive
    noise is assumed.  It is called "Latent" because the underlying function
    values are treated as latent variables.  It has a `prior` method and a
    `conditional` method.  Given a mean and covariance function the
    function :math:`f(x)` is modeled as,

    .. math::
       f(x) \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

    Use the `prior` and `conditional` methods to actually construct random
    variables representing the unknown, or latent, function whose
    distribution is the GP prior or GP conditional.  This GP implementation
    can be used to implement regression on data that is not normally
    distributed. For more information on the `prior` and `conditional` methods,
    see their docstrings.

    Parameters
    ----------
    cov_fn : pm.Covariance
        The covariance function.
    mean_fn : pm.Mean
        The mean function.

    Examples
    --------
    We can construct a LatentGP class with multiple ``feature_ndims``
    and multiple batches. Below is an example with ``feature_ndims=2``
    , ``batch_shape=2``, 10 prior samples, and 5 new samples. Notice
    that unlike PyMC3, ``given`` in ``conditional`` method is NOT optional.
    .. code:: python

        X = np.random.randn(2, 10, 2, 2)
        Xnew = np.random.randn(2, 5, 2, 2)

        # Let's define out GP model and its parameters
        mean_fn = pm.gp.mean.Zero(feature_ndims=2)
        cov_fn = pm.gp.cov.ExpQuad(1., 1., feature_ndims=2)
        gp = pm.gp.LatentGP(mean_fn, cov_fn)

        @pm.model
        def gpmodel():
            prior = yield gp.prior('prior', X)
            cond = yield gp.conditional('cond', Xnew, given={'X': X, 'f': prior})
            return cond

    
    """

    def _is_univariate(self, X):
        r"""Check if there is only one sample point"""
        return X.shape[-(self.cov_fn.feature_ndims + 1)] == 1

    def _build_prior(self, name, X, **kwargs):
        mu = self.mean_fn(X)
        cov = stabilize(self.cov_fn(X, X), noise=1e-4)
        if self._is_univariate(X):
            return Normal(
                name=name,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                **kwargs,
            )
        return MvNormal(name, loc=mu, covariance_matrix=cov, **kwargs)

    def _get_given_vals(self, given):
        r"""Get the conditional parameters"""
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
        # raise an error if the prior ``f`` is not a tensor or numpy array
        if not tf.is_tensor(f):
            try:
                f = tf.convert_to_tensor(f)
            except:
                raise ValueError("Prior `f` must be a numpy array or tensor.")

        # I am not sure about this. Do we need to worry about ``f`` being
        # sampled previously? Do we need that support?
        # if len(f.shape) > len(X.shape[:-(self.feature_ndims)]):
        #     # f is previously sampled and may contain several chains and samples
        #     # simply broadcast the shape of data to match f.
        #     X = tf.broadcast_to(X, f.shape + X.shape[-(self.feature_ndims):])
        #     Xnew = tf.broadcast_to(Xnew, f.shape + Xnew.shape[-(self.feature_ndims):])

        # We need to add an extra dimension onto ``f`` for univariate
        # distributions to make the shape consistent with ``mean_total(X)``
        if self._is_univariate(X) and len(f.shape) < len(X.shape[: -(self.feature_ndims)]):
            f = f[..., tf.newaxis]
        Kxx = cov_total(X, X)
        Kxs = self.cov_fn(X, Xnew)
        L = tf.linalg.cholesky(stabilize(Kxx))
        A = tf.linalg.solve(L, Kxs)
        # We add a `newaxis` to make the shape of mean_total(X)
        # [batch_shape, num_samples, 1] which is consistent with
        # the shape `tf.linalg.solve` accepts.
        v = tf.linalg.solve(L, (f - mean_total(X))[..., tf.newaxis])
        # Add an axis to avoid broadcasting
        mu = self.mean_fn(Xnew)[..., tf.newaxis] + tf.linalg.matrix_transpose(A) @ v
        Kss = self.cov_fn(Xnew, Xnew)
        cov = Kss - tf.linalg.matrix_transpose(A) @ A
        # Return the stabilized covarience matrix and squeeze the
        # last dimension that we added earlier.
        return tf.squeeze(mu, axis=[-1]), stabilize(cov, noise=1e-4)

    def prior(self, name, X, **kwargs):
        r"""Returns the GP prior distribution evaluated over the input locations `X`.
        
        This is the prior probability over the space
        of functions described by its mean and covariance function.

        .. math::
           f \mid X \sim \text{MvNormal}\left( \mu(X), k(X, X') \right)

        Parameters
        ----------
        name : string
            Name of the random variable
        X : tensor, array-like
            Function input values.
        **kwargs :
            Extra keyword arguments that are passed to distribution constructor.
        """
        f = self._build_prior(name, X, **kwargs)
        self.X = X
        self.f = f
        return f

    def conditional(self, name, Xnew, given, **kwargs):
        r"""Returns the conditional distribution evaluated over new input
        locations `Xnew`.
        Given a set of function values `f` that
        the GP prior was over, the conditional distribution over a
        set of new points, `f_*` is
        .. math::
           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) K(X, X)^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) K(X, X)^{-1} K(X, X_*) \right)
        Parameters
        ----------
        name : string
            Name of the random variable
        Xnew : tensor, array-like
            Function input values.
        given : dict
            Dictionary containing the observed data tensor `X` under the key "X" and
            prior samples `f` unser the key "f".
        **kwargs :
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, *givens)
        if self._is_univariate(Xnew):
            return Normal(
                name=name,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
            )
        return MvNormal(name=name, loc=mu, covariance_matrix=cov, **kwargs)
