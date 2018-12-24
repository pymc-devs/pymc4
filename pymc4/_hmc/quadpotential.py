import numpy as np
from numpy.random import normal
import scipy.linalg
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from scipy import linalg, stats
import theano

from pymc3.theanof import floatX


__all__ = ['quad_potential', 'QuadPotentialDiag', 'QuadPotentialFull',
           'QuadPotentialFullInv', 'QuadPotentialDiagAdapt', 'isquadpotential',
           'QuadPotentialLowRank']


def quad_potential(C, is_cov):
    """
    Compute a QuadPotential object from a scaling matrix.

    Parameters
    ----------
    C : arraylike, 0 <= ndim <= 2
        scaling matrix for the potential
        vector treated as diagonal matrix.
    is_cov : Boolean
        whether C is provided as a covariance matrix or hessian

    Returns
    -------
    q : Quadpotential
    """
    if issparse(C):
        if not chol_available:
            raise ImportError("Sparse mass matrices require scikits.sparse")
        elif is_cov:
            return QuadPotentialSparse(C)
        else:
            raise ValueError("Sparse precision matrices are not supported")

    partial_check_positive_definite(C)
    if C.ndim == 1:
        if is_cov:
            return QuadPotentialDiag(C)
        else:
            return QuadPotentialDiag(1. / C)
    else:
        if is_cov:
            return QuadPotentialFull(C)
        else:
            return QuadPotentialFullInv(C)


def partial_check_positive_definite(C):
    """Make a simple but partial check for Positive Definiteness."""
    if C.ndim == 1:
        d = C
    else:
        d = np.diag(C)
    i, = np.nonzero(np.logical_or(np.isnan(d), d <= 0))

    if len(i):
        raise PositiveDefiniteError(
            "Simple check failed. Diagonal contains negatives", i)


class PositiveDefiniteError(ValueError):
    def __init__(self, msg, idx):
        super(PositiveDefiniteError, self).__init__(msg)
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return ("Scaling is not positive definite: %s. Check indexes %s."
                % (self.msg, self.idx))


class QuadPotential(object):
    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        raise NotImplementedError('Abstract method')

    def energy(self, x, velocity=None):
        raise NotImplementedError('Abstract method')

    def random(self, x):
        raise NotImplementedError('Abstract method')

    def velocity_energy(self, x, v_out):
        raise NotImplementedError('Abstract method')

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning.

        This can be used by adaptive potentials to change the
        mass matrix.
        """
        pass

    def raise_ok(self, vmap=None):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Parameters
        ----------
        vmap : blocking.ArrayOrdering.vmap
            List of `VarMap`s, which are namedtuples with var, slc, shp, dtyp

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        return None

    def reset(self):
        pass


def isquadpotential(value):
    """Check whether an object might be a QuadPotential object."""
    return isinstance(value, QuadPotential)


class QuadPotentialDiagAdapt(QuadPotential):
    """Adapt a diagonal mass matrix from the sample variances."""

    def __init__(self, n, initial_mean, initial_diag=None, initial_weight=0,
                 adaptation_window=101, dtype=None):
        """Set up a diagonal mass matrix."""
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError('Initial diagonal must be one-dimensional.')
        if initial_mean.ndim != 1:
            raise ValueError('Initial mean must be one-dimensional.')
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError('Wrong shape for initial_diag: expected %s got %s'
                             % (n, len(initial_diag)))
        if len(initial_mean) != n:
            raise ValueError('Wrong shape for initial_mean: expected %s got %s'
                             % (n, len(initial_mean)))

        if dtype is None:
            dtype = theano.config.floatX

        if initial_diag is None:
            initial_diag = np.ones(n, dtype=dtype)
            initial_weight = 1

        self.dtype = dtype
        self._n = n
        self._var = np.array(initial_diag, dtype=self.dtype, copy=True)
        self._var_theano = theano.shared(self._var)
        self._stds = np.sqrt(initial_diag)
        self._inv_stds = floatX(1.) / self._stds
        self._foreground_var = _WeightedVariance(
            self._n, initial_mean, initial_diag, initial_weight, self.dtype)
        self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
        self._n_samples = 0
        self.adaptation_window = adaptation_window

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        return np.multiply(self._var, x, out=out)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * x.dot(velocity)
        return 0.5 * x.dot(self._var * x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    def random(self):
        """Draw random value from QuadPotential."""
        vals = normal(size=self._n).astype(self.dtype)
        return self._inv_stds * vals

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)
        self._var_theano.set_value(self._var)

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if not tune:
            return

        window = self.adaptation_window

        self._foreground_var.add_sample(sample, weight=1)
        self._background_var.add_sample(sample, weight=1)
        self._update_from_weightvar(self._foreground_var)

        if self._n_samples > 0 and self._n_samples % window == 0:
            self._foreground_var = self._background_var
            self._background_var = _WeightedVariance(self._n, dtype=self.dtype)

        self._n_samples += 1

    def raise_ok(self, vmap):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Parameters
        ----------
        vmap : blocking.ArrayOrdering.vmap
            List of `VarMap`s, which are namedtuples with var, slc, shp, dtyp

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        if np.any(self._stds == 0):
            name_slc = []
            tmp_hold = list(range(self._stds.size))
            for vmap_ in vmap:
                slclen = len(tmp_hold[vmap_.slc])
                for i in range(slclen):
                    name_slc.append((vmap_.var, i))
            index = np.where(self._stds == 0)[0]
            errmsg = ['Mass matrix contains zeros on the diagonal. ']
            for ii in index:
                errmsg.append('The derivative of RV `{}`.ravel()[{}]'
                              ' is zero.'.format(*name_slc[ii]))
            raise ValueError('\n'.join(errmsg))

        if np.any(~np.isfinite(self._stds)):
            name_slc = []
            tmp_hold = list(range(self._stds.size))
            for vmap_ in vmap:
                slclen = len(tmp_hold[vmap_.slc])
                for i in range(slclen):
                    name_slc.append((vmap_.var, i))
            index = np.where(~np.isfinite(self._stds))[0]
            errmsg = ['Mass matrix contains non-finite values on the diagonal. ']
            for ii in index:
                errmsg.append('The derivative of RV `{}`.ravel()[{}]'
                              ' is non-finite.'.format(*name_slc[ii]))
            raise ValueError('\n'.join(errmsg))


class QuadPotentialDiagAdaptGrad(QuadPotentialDiagAdapt):
    """Adapt a diagonal mass matrix from the variances of the gradients.

    This is experimental, and may be removed without prior deprication.
    """

    def __init__(self, *args, **kwargs):
        super(QuadPotentialDiagAdaptGrad, self).__init__(*args, **kwargs)
        self._grads1 = np.zeros(self._n, dtype=self.dtype)
        self._ngrads1 = 0
        self._grads2 = np.zeros(self._n, dtype=self.dtype)
        self._ngrads2 = 0

    def _update(self, var):
        self._var[:] = var
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)
        self._var_theano.set_value(self._var)

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if not tune:
            return

        self._grads1[:] += np.abs(grad)
        self._grads2[:] += np.abs(grad)
        self._ngrads1 += 1
        self._ngrads2 += 1

        if self._n_samples <= 150:
            super().update(sample, grad)
        else:
            self._update((self._ngrads1 / self._grads1) ** 2)

        if self._n_samples > 100 and self._n_samples % 100 == 50:
            self._ngrads1 = self._ngrads2
            self._ngrads2 = 1
            self._grads1[:] = self._grads2
            self._grads2[:] = 1


class _WeightedVariance(object):
    """Online algorithm for computing mean of variance."""

    def __init__(self, nelem, initial_mean=None, initial_variance=None,
                 initial_weight=0, dtype='d'):
        self._dtype = dtype
        self.w_sum = float(initial_weight)
        self.w_sum2 = float(initial_weight) ** 2
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype='d')
        else:
            self.mean = np.array(initial_mean, dtype='d', copy=True)
        if initial_variance is None:
            self.raw_var = np.zeros(nelem, dtype='d')
        else:
            self.raw_var = np.array(initial_variance, dtype='d', copy=True)

        self.raw_var[:] *= self.w_sum

        if self.raw_var.shape != (nelem,):
            raise ValueError('Invalid shape for initial variance.')
        if self.mean.shape != (nelem,):
            raise ValueError('Invalid shape for initial mean.')

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.w_sum += weight
        self.w_sum2 += weight * weight
        prop = weight / self.w_sum
        old_diff = x - self.mean
        self.mean[:] += prop * old_diff
        new_diff = x - self.mean
        self.raw_var[:] += weight * old_diff * new_diff

    def current_variance(self, out=None):
        if self.w_sum == 0:
            raise ValueError('Can not compute variance without samples.')
        if out is not None:
            return np.divide(self.raw_var, self.w_sum, out=out)
        else:
            return (self.raw_var / self.w_sum).astype(self._dtype)

    def current_mean(self):
        return self.mean.copy(dtype=self._dtype)


class QuadPotentialDiag(QuadPotential):
    """Quad potential using a diagonal covariance matrix."""

    def __init__(self, v, dtype=None):
        """Use a vector to represent a diagonal matrix for a covariance matrix.

        Parameters
        ----------
        v : vector, 0 <= ndim <= 1
           Diagonal of covariance matrix for the potential vector
        """
        if dtype is None:
            dtype = theano.config.floatX
        self.dtype = dtype
        v = v.astype(self.dtype)
        s = v ** .5

        self.s = s
        self.inv_s = 1. / s
        self.v = v

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        if out is not None:
            np.multiply(x, self.v, out=out)
            return
        return self.v * x

    def random(self):
        """Draw random value from QuadPotential."""
        return floatX(normal(size=self.s.shape)) * self.inv_s

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * np.dot(x, velocity)
        return .5 * x.dot(self.v * x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        np.multiply(x, self.v, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadPotentialFullInv(QuadPotential):
    """QuadPotential object for Hamiltonian calculations using inverse of covariance matrix."""

    def __init__(self, A, dtype=None):
        """Compute the lower cholesky decomposition of the potential.

        Parameters
        ----------
        A : matrix, ndim = 2
           Inverse of covariance matrix for the potential vector
        """
        if dtype is None:
            dtype = theano.config.floatX
        self.dtype = dtype
        self.L = floatX(scipy.linalg.cholesky(A, lower=True))

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        vel = scipy.linalg.cho_solve((self.L, True), x)
        if out is None:
            return vel
        out[:] = vel

    def random(self):
        """Draw random value from QuadPotential."""
        n = floatX(normal(size=self.L.shape[0]))
        return np.dot(self.L, n)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is None:
            velocity = self.velocity(x)
        return .5 * x.dot(velocity)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadPotentialFull(QuadPotential):
    """Basic QuadPotential object for Hamiltonian calculations."""

    def __init__(self, A, dtype=None):
        """Compute the lower cholesky decomposition of the potential.

        Parameters
        ----------
        A : matrix, ndim = 2
            scaling matrix for the potential vector
        """
        if dtype is None:
            dtype = theano.config.floatX
        self.dtype = dtype
        self.A = A.astype(self.dtype)
        self.L = scipy.linalg.cholesky(A, lower=True)

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        return np.dot(self.A, x, out=out)

    def random(self):
        """Draw random value from QuadPotential."""
        n = floatX(normal(size=self.L.shape[0]))
        return scipy.linalg.solve_triangular(self.L.T, n)

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is None:
            velocity = self.velocity(x)
        return .5 * x.dot(velocity)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    __call__ = random


def add_ADATv(A, v, out, diag=None, beta=0., work=None):
    """Run out = beta * out + A @ np.diag(D) @ A.T @ v"""
    if work is None:
        work = np.empty(A.shape[1])
    linalg.blas.dgemv(1., A, v, y=work, trans=1, beta=0., overwrite_y=True)
    if diag is not None:
        work *= diag
    linalg.blas.dgemv(1., A, work, y=out, beta=beta, overwrite_y=True)


class Covariance:
    def __init__(self, n_dim, n_svd, n_approx, values, grads, diag=None):
        assert n_svd <= len(values)
        assert values.shape == grads.shape

        self.values = values - values.mean(0)
        self.grads = grads - grads.mean(0)

        val_variance = self.values.var(0)
        grd_variance = self.grads.var(0)
        self._val_var = val_variance
        self._grd_var = grd_variance
        if diag == 'mean':
            self.diag = np.sqrt(val_variance/grd_variance)
        elif diag == 'values':
            self.diag = np.sqrt(val_variance)
        elif isinstance(diag, np.ndarray):
            self.diag = np.sqrt(diag)
        else:
            raise ValueError('Unknown diag approximation: %s' % diag)
        self.invsqrtdiag = 1 / np.sqrt(self.diag)
        self.values /= self.diag[None, :]
        self.grads *= self.diag[None, :]

        _, svdvals, vecs = linalg.svd(self.values, full_matrices=False)
        self.vals_eigs = 2 * np.log(svdvals[:n_svd]) - np.log(len(values))
        self.vals_vecs = vecs.T[:, :n_svd].copy()

        _, svdvals, vecs = linalg.svd(self.grads, full_matrices=False)
        self.grad_eigs = -2 * np.log(svdvals[:n_svd]) + np.log(len(grads))
        self.grad_vecs = vecs.T[:, :n_svd].copy()

        self.n_dim = n_dim
        self.n_svd = n_svd
        self.n_approx = n_approx

        if n_svd < n_dim // 3:
            center_slice = slice(n_svd // 3, None)
        else:
            center_slice = slice(2*n_svd // 3, (2 * n_dim) // 3)

        self.center = 0.5 * (
            self.grad_eigs[center_slice].mean()
            + self.vals_eigs[center_slice].mean())

        self.vals_eigs -= self.center
        self.grad_eigs -= self.center

        weight = stats.beta(0.5, 0.5).cdf(np.linspace(0, 1, n_dim))
        self.weight = 1 - weight[:n_svd]

        self._make_operators(n_approx)

    def to_dense(self):
        vecs, eigs = self.vals_vecs, self.vals_eigs
        A = (vecs * eigs * self.weight) @ vecs.T

        vecs, eigs = self.grad_vecs, self.grad_eigs
        B = (vecs * eigs * self.weight) @ vecs.T

        corr = np.exp(self.center) * linalg.expm(A + B)
        corr *= self.diag[:, None]
        corr *= self.diag[None, :]
        return corr

    def invsqrt_to_dense(self):
        assert False  # TODO This is wrong
        vecs, eigs = self.vals_vecs, self.vals_eigs
        A = (vecs * eigs * self.weight) @ vecs.T

        vecs, eigs = self.grad_vecs, self.grad_eigs
        B = (vecs * eigs * self.weight) @ vecs.T

        corr = np.exp(-0.5*self.center) * linalg.expm(-0.5*(A + B))
        corr *= self.invsqrtdiag[:, None]
        corr *= self.invsqrtdiag[None, :]
        return corr

    def matmul(self, x, out=None):
        if out is None:
            out = np.empty_like(x)

        self._matmul(x * self.diag, out)
        out *= self.diag
        return out

    def invsqrtmul(self, x, out=None):
        if out is None:
            out = np.empty_like(x)
        self._matmul_invsqrt(x, out)
        return out / self.diag

    def _make_operators(self, n_eigs, exponent=1):
        vecs1, eigs1 = self.vals_vecs, self.vals_eigs
        vecs2, eigs2 = self.grad_vecs, self.grad_eigs
        vecs1 = np.ascontiguousarray(vecs1)
        vecs2 = np.ascontiguousarray(vecs2)

        def upper_matmul(x):
            out = np.empty_like(x)
            work = np.empty(len(eigs1))
            add_ADATv(vecs1, x, out, diag=eigs1 * self.weight, beta=0.0, work=work)
            add_ADATv(vecs2, x, out, diag=eigs2 * self.weight, beta=1.0, work=work)
            return out

        upper = slinalg.LinearOperator((self.n_dim, self.n_dim), upper_matmul)
        eigs, vecs = slinalg.eigsh(upper, k=n_eigs, mode='buckling')
        self._matrix_logeigs = eigs
        eigs_exp = np.exp(eigs)
        eigs_invsqrtexp = np.exp(-0.5*eigs)

        def matmul_exp(x, out):
            work = np.empty(len(eigs))
            add_ADATv(vecs, x, out, diag=None, beta=0.0, work=work)
            add_ADATv(vecs, x, out, diag=eigs_exp, beta=-1.0, work=work)
            out += x
            out *= np.exp(self.center)

        def matmul_invsqrtexp(x, out):
            work = np.empty(len(eigs))
            add_ADATv(vecs, x, out, diag=None, beta=0.0, work=work)
            add_ADATv(vecs, x, out, diag=eigs_invsqrtexp, beta=-1.0, work=work)
            out += x
            out *= np.exp(-0.5*self.center)

        self._matmul = matmul_exp
        self._matmul_invsqrt = matmul_invsqrtexp


class QuadPotentialLowRank(object):
    def __init__(self, ndim, n_approx, diag):
        self._cov = None
        self._iter = 0
        self._ndim = ndim
        self._n_approx = n_approx
        self._diag = diag
        self._old_covs = []

        self._grad_store = []
        self._sample_store = []
        self.dtype = 'float64'

    def velocity(self, x, out=None):
        if self._cov is None:
            if out is None:
                out = np.empty_like(x)
            out[:] = x
            return out

        return self._cov.matmul(x, out=out)

    def energy(self, x, velocity=None):
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * x.dot(velocity)

    def random(self):
        rand = np.random.randn(self._ndim)
        if self._cov is None:
            return rand
        return self._cov.invsqrtmul(rand)

    def velocity_energy(self, x, v_out):
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    def raise_ok(self, *args, **kwargs):
        pass

    def update(self, sample, grad, tune):
        self._iter += 1
        if not tune:
            return

        if self._iter < 50:
            return

        renew_iters = [120, 240, 400, 850]
        if self._iter not in renew_iters:
            self._grad_store.append(grad.copy())
            self._sample_store.append(sample.copy())
            return

        n_samples = len(self._grad_store)
        samples = np.array(self._sample_store)
        grads = np.array(self._grad_store)
        self._sample_store.clear()
        self._grad_store.clear()
        if self._iter <= 160:
            n_approx = 4
        else:
            n_approx = self._n_approx
        if self._cov is not None:
            self._old_covs.append(self._cov)
        n_svd = min(self._ndim - 5, n_samples - 5)
        self._cov = Covariance(self._ndim, n_svd, n_approx,
                               samples, grads, diag=self._diag)


try:
    import sksparse.cholmod as cholmod
    chol_available = True
except ImportError:
    chol_available = False

if chol_available:
    __all__ += ['QuadPotentialSparse']

    import theano.sparse

    class QuadPotentialSparse(QuadPotential):
        def __init__(self, A):
            """Compute a sparse cholesky decomposition of the potential.

            Parameters
            ----------
            A : matrix, ndim = 2
                scaling matrix for the potential vector
            """
            self.A = A
            self.size = A.shape[0]
            self.factor = factor = cholmod.cholesky(A)
            self.d_sqrt = np.sqrt(factor.D())

        def velocity(self, x):
            """Compute the current velocity at a position in parameter space."""
            A = theano.sparse.as_sparse(self.A)
            return theano.sparse.dot(A, x)

        def random(self):
            """Draw random value from QuadPotential."""
            n = floatX(normal(size=self.size))
            n /= self.d_sqrt
            n = self.factor.solve_Lt(n)
            n = self.factor.apply_Pt(n)
            return n

        def energy(self, x):
            """Compute kinetic energy at a position in parameter space."""
            return 0.5 * x.T.dot(self.velocity(x))
