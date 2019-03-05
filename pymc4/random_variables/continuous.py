"""
PyMC4 continuous random variables.

Wraps selected tfp.distributions (listed in __all__) as pm.RandomVariables.
Implements random variables not supported by tfp as distributions.
"""

# FIXME all RandomVariable classes need docstrings
# pylint: disable=undefined-all-variable
import sys

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow import matmul, reshape, concat, transpose, zeros, sign, diag, ones, log, dtypes, expand_dims, broadcast_to, ones, scatter_update , Variable, rank, size
from tensorflow import range as tf_range
from math import pi
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape, constant_op
from tensorflow.python.ops.distributions import distribution

from .random_variable import RandomVariable


class HalfStudentT(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """
        Half student-T base distribution.

        A HalfStudentT is the absolute value of a StudentT.
        """
        return tfd.TransformedDistribution(
            distribution=tfd.StudentT(*args, **kwargs),
            bijector=tfp.bijectors.AbsoluteValue(),
            name="HalfStudentT",
        )


class LogitNormal(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """
        Logit normal base distribution.

        A LogitNormal is the standard logistic (i.e. sigmoid) of a Normal.
        """
        return tfd.TransformedDistribution(
            distribution=tfd.Normal(*args, **kwargs),
            bijector=tfp.bijectors.Sigmoid(),
            name="LogitNormal",
        )


    # what is the right name?
class StdSkewNormal(tfp.distributions.Distribution):
    # STATISTICAL APPLICATIONS OF THE MULTIVARIATE SKEW-NORMAL DISTRIBUTION
    # A. Azzalini
    # https://arxiv.org/pdf/0911.2093.pdf
    # paramed by alpha and omega
    def __init__(self,
               corr,  # correlation (sq.) m x m matrix (Omega in Azzalini)
               skew,  # skew/shape vector as m x 1 matrix (alpha in Azzalini)
               validate_args=False,
               allow_nan_stats=True,
               name="StdSkewNormal"):
        """
        Construct Skew Normal distributions.
        The parameters `loc` and `scale` must be shaped in a way that supports
        broadcasting (e.g. `loc + scale` is a valid operation).
        Args:
          loc: Floating point tensor; the means of the distribution(s).
          scale: Floating point tensor; the stddevs of the distribution(s).
            Must contain only positive values.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is raised
            if one or more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.
        Raises:
          TypeError: if `loc` and `scale` have different `dtype`.
        """
        # TODO: if skew=None, make a 0 skew.
        # is corr foreced to equal 1 when univariate?
        parameters = dict(locals())
        #with ops.name_scope(name, values=[loc, scale]) as name:
        #  with ops.control_dependencies([check_ops.assert_positive(scale)] if
        #                                validate_args else []):
        #   self._corr = array_ops.identity(corr, name="correlation")
        #    self._skew = array_ops.identity(skew, name="skew")
        #    check_ops.assert_same_float_dtype([self._loc, self._scale])
        
        
        self._corr = array_ops.identity(corr, name="correlation")
        self._skew = array_ops.identity(skew, name="skew")
        # assume matrix
        if self._corr.ndim == 2:
            self._corr = expand_dims(self._corr, 0)
        # assume vec 
        if self._skew.ndim == 1:
            self._skew = expand_dims(self._skew, 0)
        
                
        super(StdSkewNormal, self).__init__(
            dtype=self._corr.dtype,
            reparameterization_type=distribution.FULLY_REPARAMETERIZED, # ?
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._corr, self._skew],
            name=name)
    
    @property
    def corr(self):
        """correlation matrix"""
        return self._corr
    
    @property
    def Omega(self):
        shape = concat([self.batch_shape_tensor(), array_ops.shape(self.corr)[-2:]], 0)
        return broadcast_to(self.corr, shape)
    
    # why doesn't TFP Distribution do this?
    @property
    def skew(self):
        """skew vector"""
        return self._skew
    
    @property
    def alpha(self):
        alpha = expand_dims(self.skew, -1) # makes a (col., n x 1) vector 
        shape = concat([self.batch_shape_tensor(), array_ops.shape(alpha)[-2:]], 0)
        return broadcast_to(alpha, shape)
    
    def _mean(self):
        return (2./pi)**(.5) * self.get_delta()
    
    def _variance(self):
        mu = self.mean()
        return self.Omega - matmul(mu, mu, transpose_b=True)
    
    #@property # is property a good idea here, since there are computations involved?
    #def delta(self):
     #   return get_delta if not calculated
    
    def get_delta(self):
        # TODO: check correct when broadcasted
        Omegaalpha = self.Omega @ self.alpha
        return Omegaalpha @ (1 + matmul(self.alpha, Omegaalpha, transpose_a=True))**(-.5)
    
    #@property
    def get_Omega_star(self):
        """correlation matrix of underlying normal distribution"""
        delta = self.get_delta()
        Omega = self.Omega
        one_shp = array_ops.shape(delta)
        # one_shp[-2] = 1 doesn't work
        # https://github.com/tensorflow/tensorflow/issues/14132
        # so instead, ...
        one_shp = concat([one_shp[:-2] , [1,1] ], 0)
        one = ones(one_shp)
        col0 = concat([one, delta], -2)
        
        delta_dim = tf_range(rank(delta))
        delta_dim = concat([delta_dim[:-2], [delta_dim[-1]], [delta_dim[-2]]], 0)
        col1 = concat([transpose(delta, perm=delta_dim ), Omega], -2)
        return concat( [col0, col1], -1)
        
    def _batch_shape_tensor(self): 
        # In other words, take the last 1 or 2 dims ( take as col. vector or matrix)
        # and 'broadcast' the leading dims
        return array_ops.broadcast_dynamic_shape(
            array_ops.shape(self._corr)[:-2],
            array_ops.shape(self._skew)[:-1]) # Tensor

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
            self._corr.get_shape()[:-2],
            self._skew.get_shape()[:-1])  # TensorShape
    
    def _event_shape_tensor(self):
        return constant_op.constant([array_ops.shape(self.skew)[-1]], dtype=dtypes.int32)

    def _event_shape(self):
        return tensor_shape.TensorShape([self.skew.get_shape()[-1].value])
    
    def _sample_n(self, n, seed=None):
        kplus1 = self.event_shape[0] + 1
        N = tfd.MultivariateNormalFullCovariance(loc=zeros(kplus1), covariance_matrix=self.get_Omega_star())
        X0X = N.sample(sample_shape=n, seed=seed)
        X0 = X0X[..., :1]
        X =  X0X[..., 1:]
        signX0 = sign(X0)
        return signX0 * X
        
    def get_prob_args(self, x):
        k = self.event_shape[0]
        N0O = tfd.MultivariateNormalFullCovariance(loc=zeros(k), covariance_matrix=self.Omega)
        N01 = tfd.Normal(loc=0, scale=1)
        
        not_sample_rank = size(self.batch_shape) + size(self.event_shape)
        sample_shp = array_ops.shape(x)[:-not_sample_rank]
        alpha_shp = concat([sample_shp, array_ops.shape(self.alpha)], 0)
        #assert(alpha_dims[-1] == 1) how would this work in tf??
        alpha = broadcast_to(self.alpha, alpha_shp)
        return alpha, N0O, N01

    def _prob(self, x):
        alpha, N0O, N01 = self.get_prob_args(x)
        N01cdf = N01.cdf(matmul(alpha, x[..., None], transpose_a=True))[..., 0, 0]
        return 2. * N0O.prob(x) * N01cdf
    
    def _log_prob(self, x):
        alpha, N0O, N01 = self.get_prob_args(x)
        N01cdf = N01.log_cdf(matmul(alpha, x[..., None], transpose_a=True))[..., 0, 0]
        return log(2.) + N0O.log_prob(x) + N01cdf

    
class Weibull(RandomVariable):
    def _base_dist(self, *args, **kwargs):
        """
        Weibull base distribution.

        The inverse of the Weibull bijector applied to a U[0, 1] random
        variable gives a Weibull-distributed random variable.
        """
        return tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=0.0, high=1.0),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Weibull(*args, **kwargs)),
            name="Weibull",
        )


# Random variables that tfp supports as distributions. We wrap these
# distributions as random variables. Names must match tfp.distributions names
# exactly.
tfp_supported = [
    "Beta",
    "Cauchy",
    "Chi2",
    "Exponential",
    "Gamma",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "InverseGaussian",
    "Kumaraswamy",
    "Laplace",
    "LogNormal",
    "Logistic",
    "Normal",
    "Pareto",
    "StudentT",
    "Triangular",
    "Uniform",
    "VonMises",
]

# Programmatically wrap tfp.distribtions into pm.RandomVariables
for dist_name in tfp_supported:
    setattr(
        sys.modules[__name__],
        dist_name,
        type(dist_name, (RandomVariable,), {"_base_dist": getattr(tfd, dist_name)}),
    )
