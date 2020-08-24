"""An Interface for creating Gaussian Process Models in PyMC4."""

from typing import Union, Optional, Any

import tensorflow as tf

import pymc4 as pm
from .mean import Mean, Zero
from .cov import Covariance
from ..distributions import MvNormalCholesky, MvNormal, Normal, ContinuousDistribution
from .util import stabilize, ArrayLike, TfTensor, FreeRV


NameType = Union[str, int]


__all__ = ["LatentGP", "MarginalGP"]


class BaseGP:
    def __init__(self, cov_fn: Covariance, mean_fn: Mean = Zero(1)):
        self.feature_ndims = max(mean_fn.feature_ndims, cov_fn.feature_ndims)
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def _is_univariate(self, X: ArrayLike) -> bool:
        r"""Check if there is only one sample point."""
        return X.shape[-(self.feature_ndims + 1)] == 1

    def prior(
        self,
        name: NameType,
        X: ArrayLike,
        *,
        reparametrize=True,
        jitter: Optional[float] = None,
        **kwargs,
    ) -> ContinuousDistribution:
        raise NotImplementedError

    def conditional(
        self,
        name: NameType,
        Xnew: ArrayLike,
        given: dict,
        *,
        reparametrize: bool = True,
        jitter: Optional[float] = None,
        **kwargs,
    ) -> ContinuousDistribution:
        raise NotImplementedError

    def predict(self, Xnew: ArrayLike, **kwargs) -> TfTensor:
        raise NotImplementedError

    def marginal_likelihood(
        self,
        name: NameType,
        X: ArrayLike,
        *,
        reparametrize=True,
        jitter: Optional[float] = None,
        **kwargs,
    ) -> ContinuousDistribution:
        raise NotImplementedError


class LatentGP(BaseGP):
    r"""
    Latent Gaussian process.

    The `gp.LatentGP` class is a direct implementation of a GP.  No additive noise is assumed.
    It is called "Latent" because the underlying function values are treated as latent variables.
    It has a `prior` method and a `conditional` method.  Given a mean and covariance function the
    function :math:`f(x)` is modeled as,

    .. math::
       f(x) \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

    Use the `prior` and `conditional` methods to actually construct random variables representing
    the unknown, or latent, function whose distribution is the GP prior or GP conditional.
    This GP implementation can be used to implement regression on data that is not normally
    distributed. For more information on the `prior` and `conditional` methods,
    see their docstrings.

    Parameters
    ----------
    cov_fn : pm.gp.Covariance
        The covariance function.
    mean_fn : pm.gp.Mean, optional
        The mean function. Defaults to ``Zero`` mean

    Examples
    --------
    We can construct a LatentGP class with multiple ``feature_ndims`` and multiple batches.
    Below is an example with ``feature_ndims=2`` , ``batch_shape=2``, 10 prior samples,
    and 5 new samples. Notice that unlike PyMC3, ``given`` in ``conditional`` method is
    NOT optional.

    >>> import numpy as np
    >>> import pymc4 as pm
    >>> X = np.random.randn(2, 10, 2, 2)
    >>> Xnew = np.random.randn(2, 5, 2, 2)
    >>> # Let's define out GP model and its parameters
    ... mean_fn = pm.gp.mean.Zero(feature_ndims=2)
    >>> cov_fn = pm.gp.cov.ExpQuad(1., 1., feature_ndims=2)
    >>> gp = pm.gp.LatentGP(mean_fn, cov_fn)
    >>> @pm.model
    ... def gpmodel():
    ...     f = yield gp.prior('f', X)
    ...     fcond = yield gp.conditional('fcond', Xnew, given={'X': X, 'f': f})
    ...     return fcond
    """

    def _build_prior(self, name, X: ArrayLike, jitter: Optional[float] = None) -> tuple:
        mu = self.mean_fn(X)
        cov = stabilize(self.cov_fn(X, X), shift=jitter)
        return mu, cov

    def _get_given_vals(self, given: dict) -> tuple:
        r"""Get the conditional parameters."""
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
            # XXX: Maybe we can add this feature later
            # X, f = self.X, self.f
            raise ValueError(
                "given must contain 'X' and 'f' keys. found {} only.".format(list(given.keys()))
            )
        return X, f, cov_total, mean_total

    def _build_conditional(
        self,
        Xnew: ArrayLike,
        X: ArrayLike,
        f: FreeRV,
        cov_total: Covariance,
        mean_total: Mean,
        jitter: Optional[float] = None,
    ) -> tuple:
        # We need to add an extra dimension onto ``f`` for univariate
        # distributions to make the shape consistent with ``mean_total(X)``
        if self._is_univariate(X) and len(f.shape) < len(X.shape[: -(self.feature_ndims)]):
            f = tf.expand_dims(f, -1)
        Kxx = cov_total(X, X)
        Kxs = self.cov_fn(X, Xnew)
        L = tf.linalg.cholesky(stabilize(Kxx, shift=jitter))
        A = tf.linalg.triangular_solve(L, Kxs, lower=True)
        # We add a `newaxis` to make the shape of mean_total(X)
        # [batch_shape, num_samples, 1] which is consistent with
        # the shape `tf.linalg.solve` accepts.
        v = tf.linalg.triangular_solve(L, tf.expand_dims((f - mean_total(X)), -1), lower=True)
        # Add an axis to avoid right align broadcasting
        mu = tf.expand_dims(self.mean_fn(Xnew), -1) + tf.linalg.matmul(A, v, transpose_a=True)
        Kss = self.cov_fn(Xnew, Xnew)
        cov = Kss - tf.linalg.matmul(A, A, transpose_a=True)
        # Return the stabilized covariance matrix and squeeze the
        # last dimension that we added earlier.
        return tf.squeeze(mu, axis=[-1]), stabilize(cov, shift=jitter)

    def prior(
        self,
        name: NameType,
        X: ArrayLike,
        *,
        reparametrize: bool = True,
        jitter: Optional[float] = None,
        **kwargs,
    ) -> ContinuousDistribution:
        r"""
        Evaluate the GP prior distribution over the input locations `X`.

        This is the prior probability over the space
        of functions described by its mean and covariance function.

        .. math::
           f \mid X \sim \text{MvNormal}\left( \mu(X), k(X, X') \right)

        Parameters
        ----------
        name : string
            Name of the random variable.
        X : array_like
            Function input values.
        reparametrize : bool, optional
            If ``True``, ``MvNormalCholesky`` distribution is returned instead
            of ``MvNormal``. (default=``True``)
        jitter : float, optional
            The amount of diagonal shift to add to the covariance matrix to
            avoid cholesky decomposition failures. Defaults to 1e-4 for float32
            tensors and 1e-6 for float64 tensors.

        Other Parameters
        ----------------
        **kwargs :
            Extra keyword arguments that are passed to the distribution constructor.

        Returns
        -------
        f : tensorflow.Tensor
            Gaussian Process prior distribution.

        Examples
        --------
        >>> import pymc4 as pm
        >>> import numpy as np
        >>> X = np.linspace(0, 1, 10)[..., np.newaxis]
        >>> X = X.astype('float32')
        >>> cov_fn = pm.gp.cov.ExpQuad(amplitude=1., length_scale=1.)
        >>> gp = pm.gp.LatentGP(cov_fn=cov_fn)
        >>> @pm.model
        ... def gp_model():
        ...     f = yield gp.prior('f', X)
        >>> model = gp_model()
        >>> trace = pm.sample(model, num_samples=10, burn_in=10)
        """
        mu, cov = self._build_prior(name, X, jitter=jitter)
        if self._is_univariate(X):
            return Normal(
                name=name,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                **kwargs,
            )
        if reparametrize:
            chol_factor = tf.linalg.cholesky(cov)
            return MvNormalCholesky(name, loc=mu, scale_tril=chol_factor, **kwargs)
        return MvNormal(name, loc=mu, covariance_matrix=cov, **kwargs)

    def conditional(
        self,
        name: NameType,
        Xnew: ArrayLike,
        given: dict,
        *,
        reparametrize: bool = True,
        jitter: Optional[float] = None,
        **kwargs,
    ) -> ContinuousDistribution:
        r"""
        Evaluate the conditional distribution evaluated over new input locations `Xnew`.

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
        Xnew : array_like
            Function input values.
        given : dict
            Dictionary containing the observed data tensor `X` under the key "X" and
            prior random variable `f` under the key "f".
        reparametrize : bool, optional
            If ``True``, ``MvNormalCholesky`` distribution is returned instead
            of ``MvNormal``. (default=``True``)
        jitter : float, optional
            The amount of diagonal shift to add to the covariance matrix to
            avoid cholesky decomposition failures. Defaults to 1e-4 for float32
            tensors and 1e-6 for float64 tensors.

        Other Parameters
        ----------------
        **kwargs :
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.

        Returns
        -------
        fcond : tensorflow.Tensor
            Gaussian Process Conditional Distribution

        Examples
        --------
        >>> import pymc4 as pm
        >>> import numpy as np
        >>> X = np.linspace(0, 1, 10)[..., np.newaxis]
        >>> Xnew = np.linspace(0, 1, 50)[..., np.newaxis]
        >>> X, Xnew = X.astype('float32'), Xnew.astype('float32')
        >>> cov_fn = pm.gp.cov.ExpQuad(amplitude=1., length_scale=1.)
        >>> gp = pm.gp.LatentGP(cov_fn=cov_fn)
        >>> @pm.model
        ... def gp_model():
        ...     f = yield gp.prior('f', X, jitter=1e-3)
        ...     fcond = yield gp.conditional('fcond', Xnew, given={'f': f, 'X': X}, jitter=1e-3)
        >>> model = gp_model()
        >>> trace = pm.sample(model, num_samples=10, burn_in=10)
        """
        X, f, cov_total, mean_total = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, X, f, cov_total, mean_total, jitter=jitter)
        if self._is_univariate(Xnew):
            return Normal(
                name=name,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                **kwargs,
            )
        if reparametrize:
            chol_factor = tf.linalg.cholesky(cov)
            return MvNormalCholesky(name, loc=mu, scale_tril=chol_factor, **kwargs)
        return MvNormal(name=name, loc=mu, covariance_matrix=cov, **kwargs)


class MarginalGP(BaseGP):
    r"""
    Marginal Gaussian Process Model.

    The `gp.MarginalGP` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed.  For more
    information on the `prior` and `conditional` methods, see their docstrings.

    Parameters
    ----------
    cov_fn : Covariance
        The covariance function.
    mean_fn : Mean
        The mean function. Defaults to zero.

    Examples
    --------
    >>> import numpy as np
    >>> import pymc4 as pm
    >>> np.random.seed(42)
    >>> X = np.linspace(0, 1, 10)[:, np.newaxis]
    >>> y = np.random.rand(X.shape[0])
    >>> Xnew = np.linspace(-1, 2, 50)[:, np.newaxis]
    >>> X, y = X.astype(np.float32), y.astype(np.float32)
    >>> Xnew = Xnew.astype(np.float32)
    >>> @pm.model
    ... def model():
    ...     cov_fn = pm.gp.cov.ExpQuad(length_scale=0.1)
    ...     gp = pm.gp.MarginalGP(cov_fn=cov_fn)
    ...     sigma = yield pm.HalfCauchy("sigma", scale=3)
    ...     y_ = yield gp.marginal_likelihood("y_", X=X, y=y, noise=sigma)
    ...     y_pred = yield gp.conditional("y_pred", Xnew=Xnew)
    ...
    >>> gpmodel = model()
    >>> trace = pm.sample(gpmodel, num_samples=10)
    """

    def _build_marginal_likelihood(
        self, X: ArrayLike, noise: Covariance, jitter: Optional[float] = None
    ) -> tuple:
        mu = self.mean_fn(X)
        Kxx = self.cov_fn(X, X)
        Knx = noise(X, X)
        cov = Kxx + Knx
        return mu, stabilize(cov, shift=jitter)

    def marginal_likelihood(  # type: ignore
        self,
        name: NameType,
        X: ArrayLike,
        y: ArrayLike,
        noise: Union[ArrayLike, Covariance],
        is_observed: bool = True,
        *,
        reparametrize: bool = True,
        jitter: Optional[float] = None,
        **kwargs,
    ) -> ContinuousDistribution:
        r"""
        Marginal Likelihood of the GP.

        Returns the marginal likelihood distribution, given the input
        locations `X`, the data `y`, and noise `noise`. This is integral
        over the product of the GP prior and a normal likelihood.

        .. math::
           y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df

        Parameters
        ----------
        name : string
            Name of the random variable.
        X : array_like
            Function input values. If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y : array_like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.
        noise : {float, array_like, Covariance}
            Standard deviation of the Gaussian noise. Can also be a Covariance for
            non-white noise.
        is_observed : bool, optional
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        reparametrize : bool, optional
            If ``True``, ``MvNormalCholesky`` distribution is returned instead
            of ``MvNormal``. (default=``True``)
        jitter : float, optional
            The amount of diagonal shift to add to the covariance matrix to
            avoid cholesky decomposition failures. Defaults to 1e-4 for float32
            tensors and 1e-6 for float64 tensors.


        Other Parameters
        ----------------
        **kwargs :
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.

        Examples
        --------
        >>> import numpy as np
        >>> import pymc4 as pm
        >>> from pymc4.gp.cov import WhiteNoise, Constant
        >>> from pymc4.gp import MarginalGP
        >>> np.random.seed(42)

        To get the ``marginal_likelihood`` of the MarginalGP over some data
        ``X`` with labels ``y``, use:

        >>> k = Constant(1.)
        >>> noise = 1e-2
        >>> X = np.linspace(0, 1, 10)[:, np.newaxis].astype(np.float32)
        >>> y = np.random.rand(X.shape[0]).astype(np.float32)
        >>> gp = MarginalGP(cov_fn=k)
        >>> y_ = gp.marginal_likelihood("y_", X, y, noise=noise)

        You can also pass a covariance object as noise to the `marginal_likelihood`
        method:

        >>> noise = WhiteNoise(1e-2)
        >>> y_ = gp.marginal_likelihood("y_", X, y, noise=noise)

        To run ADVI on the ``MarginalGP`` model, use:

        >>> from pymc4.gp.cov import ExpQuad
        >>> @pm.model
        ... def model():
        ...     ls = yield pm.HalfCauchy("ls", scale=5.)
        ...     noise = yield pm.Beta("noise", 1., 1.)
        ...     k = ExpQuad(length_scale=ls)
        ...     gp = MarginalGP(cov_fn=k)
        ...     y_ = yield gp.marginal_likelihood("y_", X, y, noise=noise)
        ...     return y_
        ...
        >>> advifit = pm.fit(model())

        Once you have fit your model, you can sample using:

        >>> samples = advifit[0].sample(100)

        If ``y`` is not the observed data, pass ``is_observed=False`` in the
        marginal likelihood method:

        >>> y_ = gp.marginal_likelihood("y_", X, y, noise=noise, is_observed=False)

        By default, some ``jitter`` is added to ensure Cholesky Decomposition passes.
        This behaviour can be turned off by passing ``jitter=0`` in the marginal
        likehood method:

        >>> y_ = gp.marginal_likelihood("y_", X, y, noise=noise, jitter=0)

        Notes
        -----
        As ``noise`` behaves exactly as ``jitter``, it is recommended to set ``jitter=False``
        to avoid adding extra noise.
        """

        if not isinstance(noise, Covariance):
            noise = pm.gp.cov.WhiteNoise(noise)
        mu, cov = self._build_marginal_likelihood(X, noise, jitter=jitter)
        self.X = X
        self.y = y
        self.noise = noise
        if self._is_univariate(X):
            if is_observed:
                return Normal(
                    name=name,
                    loc=tf.squeeze(mu, axis=[-1]),
                    scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                    observed=y,
                    **kwargs,
                )
            return Normal(
                name=name,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                **kwargs,
            )
        if reparametrize:
            chol_factor = tf.linalg.cholesky(cov)
            if is_observed:
                return MvNormalCholesky(name, loc=mu, scale_tril=chol_factor, observed=y, **kwargs)
            return MvNormalCholesky(name, loc=mu, scale_tril=chol_factor, **kwargs)
        if is_observed:
            return MvNormal(name, loc=mu, covariance_matrix=cov, observed=y, **kwargs)
        return MvNormal(name, loc=mu, covariance_matrix=cov, **kwargs)

    def _get_given_vals(self, given: Optional[dict]) -> tuple:
        if given is None:
            given = {}

        if "gp" in given:
            cov_total = given["gp"].cov_fn
            mean_total = given["gp"].mean_fn
        else:
            cov_total = self.cov_fn
            mean_total = self.mean_fn
        if all(val in given for val in ["X", "y", "noise"]):
            X, y, noise = given["X"], given["y"], given["noise"]
            if not isinstance(noise, Covariance):
                noise = pm.gp.cov.WhiteNoise(noise)
        else:
            X, y, noise = self.X, self.y, self.noise
        return X, y, noise, cov_total, mean_total

    def _build_conditional(  # type: ignore
        self,
        Xnew: ArrayLike,
        pred_noise: bool,
        diag: bool,
        X: ArrayLike,
        y: ArrayLike,
        noise: Covariance,
        cov_total: Covariance,
        mean_total: Mean,
        jitter: Optional[float] = None,
    ):
        if self._is_univariate(X) and len(y.shape) < len(X.shape[: -(self.feature_ndims)]):
            y = tf.expand_dims(y, -1)
        Kxx = cov_total(X, X)
        Kxs = self.cov_fn(X, Xnew)
        Knx = noise(X, X)
        rxx = tf.expand_dims(y - mean_total(X), -1)
        L = tf.linalg.cholesky(stabilize(Kxx, shift=jitter) + Knx)
        A = tf.linalg.triangular_solve(L, Kxs, lower=True)
        v = tf.linalg.triangular_solve(L, rxx, lower=True)
        mu = tf.expand_dims(self.mean_fn(Xnew), -1) + tf.linalg.matmul(A, v, transpose_a=True)
        if diag:
            Kss = self.cov_fn(Xnew, Xnew, diag=True)
            var = Kss - tf.reduce_sum(tf.square(A), axis=0)
            if pred_noise:
                var += noise(Xnew, Xnew, diag=True)
            return tf.squeeze(mu, axis=[-1]), var
        else:
            Kss = self.cov_fn(Xnew, Xnew)
            cov = Kss - tf.linalg.matmul(A, A, transpose_a=True)
            if pred_noise:
                cov += noise(Xnew, Xnew)
            return tf.squeeze(mu, axis=[-1]), stabilize(cov, shift=jitter)

    def conditional(  # type: ignore
        self,
        name: NameType,
        Xnew: ArrayLike,
        pred_noise: bool = False,
        given: dict = None,
        *,
        reparametrize: bool = True,
        jitter: Optional[float] = None,
        **kwargs,
    ):
        r"""
        Conditional Distribution of the GP.

        Returns the conditional distribution evaluated over new input
        locations `Xnew`.
        Given a set of function values `f` that the GP prior was over, the
        conditional distribution over a set of new points, `f_*` is:

        .. math::
           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} K(X, X_*) \right)

        Parameters
        ----------
        name : NameType
            Name of the random variable
        Xnew : array_like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise : bool, optional
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given : dict, optional
            Can optionally take as key value pairs: `X`, `y`, `noise`,
            and `gp`.
        reparametrize : bool, optional
            If ``True``, ``MvNormalCholesky`` distribution is returned instead
            of ``MvNormal``. (default=``True``)
        jitter : float, optional
            The amount of diagonal shift to add to the covariance matrix to
            avoid cholesky decomposition failures. Defaults to 1e-4 for float32
            tensors and 1e-6 for float64 tensors.

        Other Parameters
        ----------------
        **kwargs :
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.

        Examples
        --------
        You can use conditional method to get the conditional distribution
        over the new data points. This distribution can be used to predict
        over the new data points:

        >>> import numpy as np
        >>> import pymc4 as pm
        >>> from pymc4.gp.cov import WhiteNoise, Constant
        >>> from pymc4.gp import MarginalGP
        >>> np.random.seed(42)
        >>> k = Constant(1.)
        >>> noise = 1e-2
        >>> X = np.linspace(0, 1, 10)[:, np.newaxis].astype(np.float32)
        >>> y = np.random.rand(X.shape[0]).astype(np.float32)
        >>> Xnew = np.linspace(0, 1, 100)[:, np.newaxis].astype(np.float32)
        >>> gp = MarginalGP(cov_fn=k)
        >>> y_pred = gp.conditional("y_pred", Xnew, given={"X": X, "y": y, "noise": noise})

        If you have a ``marginal_likelihood`` distribution fit on some data, you need
        not provide the ``given`` dictionary to the ``conditional`` method:

        >>> y_ = gp.marginal_likelihood("y_", X, y, noise=noise)
        >>> y_pred = gp.conditional("y_pred", Xnew) # no need to pass given

        To add noise in the conditional distribution, use:

        >>> y_pred = gp.conditional("y_pred", Xnew, pred_noise=True)

        To avoid reparametrizing to the ``MvNormalCholesky`` distribution, use:

        >>> y_pred = gp.conditional("y_pred", Xnew, reparametrize=False)

        This may be numerically instable and slow but helps avoid cholesky decomposition
        which is very suseptable to fail on large data and ``float32`` datatype.

        To run ADVI approximation, create a GP model using ``pm.model``. As quality of
        ADVI fit differs with respect to the datatype used, it is better to use ``float64``
        as shown below for stable performance:

        >>> from pymc4.gp.cov import ExpQuad
        >>> @pm.model
        ... def model():
        ...     ls = yield pm.HalfCauchy("ls", scale=5.)
        ...     noise = yield pm.Beta("noise", 1., 1.)
        ...     k = ExpQuad(length_scale=ls) + WhiteNoise(1e-2)
        ...     gp = MarginalGP(cov_fn=k)
        ...     y_ = yield gp.marginal_likelihood("y_", X, y, noise=noise)
        ...     y_pred = yield gp.conditional("y_pred", Xnew, jitter=1e-3)
        ...     return y_pred
        ...
        >>> advifit = pm.fit(model())
        >>> samples = advifit[0].sample(100)

        Stack multiple GPs using either a batch of inputs or using a batch of
        distribution (priors). For example, to stack 2 GPs with different kernels, use:

        >>> k = Constant([1., 2.])
        >>> noise = 1e-2
        >>> X = np.linspace(0, 1, 10)[:, np.newaxis].astype(np.float32)
        >>> y = np.random.rand(X.shape[0]).astype(np.float32)
        >>> Xnew = np.linspace(0, 1, 100)[:, np.newaxis].astype(np.float32)
        >>> gp = MarginalGP(cov_fn=k)
        >>> y_pred = gp.conditional("y_pred", Xnew, given={"X": X, "y": y, "noise": noise})
        >>> y_pred.batch_shape
        TensorShape([2])
        """

        X, y, noise, cov_total, mean_total = self._get_given_vals(given)
        mu, cov = self._build_conditional(
            Xnew, pred_noise, False, X, y, noise, cov_total, mean_total
        )
        if self._is_univariate(Xnew):
            return Normal(
                name=name,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                **kwargs,
            )
        if reparametrize:
            chol_factor = tf.linalg.cholesky(cov)
            return MvNormalCholesky(name, loc=mu, scale_tril=chol_factor, **kwargs)
        return MvNormal(name, loc=mu, covariance_matrix=cov, **kwargs)

    def predict(
        self,
        Xnew: ArrayLike,
        diag: bool = False,
        pred_noise: bool = False,
        given: Optional[dict] = None,
        sample_shape: Any = (),
        to_numpy: bool = False,
        *,
        reparametrize: bool = True,
        **kwargs,
    ) -> tuple:
        r"""
        Get the prediction vector at new data points.

        Returns the samples from the conditional distribution as tensorflow tensors or
        numpy arrays.

        Parameters
        ----------
        Xnew : array-like
            Function input values. If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag : bool, optional
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise : bool, optional
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given : dict, optional
            Same as `conditional` method.
        sample_shape : {tuple, TensorShape, ArrayLike}
            The number of prediction vectors to be drawn from the
            conditional distribution.
        to_numpy : bool, optional
            Returns `mu` and `cov` as numpy array instead of tensorflow tensors.
        reparametrize : bool, optional
            If ``True``, samples from a ``MvNormalCholesky`` distribution, else
            samples from a ``MvNormal`` distribution. (default=``True``)

        Other Parameters
        ----------------
        **kwargs :
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.

        Examples
        --------
        >>> import numpy as np
        >>> import pymc4 as pm
        >>> from pymc4.gp.cov import WhiteNoise, Constant
        >>> from pymc4.gp import MarginalGP
        >>> np.random.seed(42)

        To sample from the conditional distribution, use:

        >>> k = Constant(1.)
        >>> noise = 1e-4
        >>> X = np.linspace(0, 1, 10)[:, np.newaxis].astype(np.float32)
        >>> y = np.random.randn(X.shape[0]).astype(np.float32)
        >>> Xnew = np.linspace(0, 1, 100)[:, np.newaxis].astype(np.float32)
        >>> gp = MarginalGP(cov_fn=k)
        >>> y_pred = gp.predict(Xnew, given={"X": X, "y": y, "noise": noise})
        >>> y_pred.shape
        TensorShape([100])

        To get multiple prediction vectors (samples from the conditional distribution)
        at once, pass the `sample_shape`` parameter:

        >>> y_pred = gp.predict(Xnew, given={"X": X, "y": y, "noise": noise}, sample_shape=20)
        >>> y_pred.shape
        TensorShape([20, 100])

        To get a numpy array instead of tensorflow tensor, use:

        >>> y_pred = gp.predict(Xnew, given={"X": X, "y": y, "noise": noise}, to_numpy=True)

        If only the diagonal of the covariance matrix is desired, use:

        >>> y_pred = gp.predict(Xnew, given={"X": X, "y": y, "noise": noise}, diag=True)
        """
        if given is None:
            given = {}
        mu, cov = self.predictt(Xnew, diag, pred_noise, given)
        if self._is_univariate(Xnew):
            samples = Normal(
                name=None,
                loc=tf.squeeze(mu, axis=[-1]),
                scale=tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2])),
                **kwargs,
            ).sample(sample_shape)
            return samples.numpy() if to_numpy else samples
        if diag:
            samples = Normal(None, loc=mu, scale=cov, **kwargs).sample(sample_shape=sample_shape)
            return samples.numpy() if to_numpy else samples
        if reparametrize:
            chol_factor = tf.linalg.cholesky(cov)
            samples = MvNormalCholesky(None, loc=mu, scale_tril=chol_factor, **kwargs).sample(
                sample_shape=sample_shape
            )
            return samples.numpy() if to_numpy else samples
        samples = MvNormal(None, loc=mu, covariance_matrix=cov, **kwargs).sample(
            sample_shape=sample_shape
        )
        return samples.numpy() if to_numpy else samples

    def predictt(
        self,
        Xnew: ArrayLike,
        diag: bool = False,
        pred_noise: bool = False,
        given: Optional[dict] = None,
        to_numpy: bool = False,
    ) -> tuple:
        r"""
        Get point predictions and covariance matrix.

        Returns the mean vector and covariance matrix of the conditional
        distribution as tensorflow tensors or numpy arrays. The mean vector
        are point predictions where the conditional probability is maximum.
        The covariance matrix can be used to quantify the uncertainties in the
        predictions.

        Parameters
        ----------
        Xnew: array_like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag: bool, optional
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool, optional
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict, optional
            Same as `conditional` method.
        to_numpy : bool, optional
            Returns `mu` and `cov` as numpy array instead of tensorflow tensors.

        Examples
        --------
        >>> import numpy as np
        >>> import pymc4 as pm
        >>> from pymc4.gp.cov import ExpQuad
        >>> from pymc4.gp import MarginalGP
        >>> np.random.seed(42)

        To get the point predictions from conditional distribution, use:

        >>> k = ExpQuad(np.array(1.))
        >>> noise = np.array(1e-8)
        >>> X = np.linspace(0, 1, 10)[:, np.newaxis]
        >>> Xnew = np.linspace(0, 1, 20)[:, np.newaxis]
        >>> y = np.random.randn(X.shape[0])
        >>> gp = MarginalGP(cov_fn=k)
        >>> mu, cov = gp.predictt(X, given={"X": X, "y": y, "noise": noise})
        >>> print(tf.reduce_mean((y - mu)**2))
        tf.Tensor(0.45554812435872166, shape=(), dtype=float64)

        To only evaluate the diagonal of the covariance matrix, use:

        >>> mu, cov = gp.predictt(Xnew, given={"X": X, "y": y, "noise": noise}, diag=True)
        """
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, pred_noise, diag, *givens)
        if self._is_univariate(Xnew):
            mu, cov = tf.squeeze(mu, axis=[-1]), tf.math.sqrt(tf.squeeze(cov, axis=[-1, -2]))
        return (mu.numpy(), cov.numpy()) if to_numpy else (mu, cov)
