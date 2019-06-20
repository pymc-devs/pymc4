from ..base.distribution import Distribution


class BackendDistribution(Distribution):
    """
    BackendDistribution for Tensorflow distributions
    """
    __doc__ = Distribution.__doc__ + __doc__

    def sample(self, *args, **kwargs):
        """
        Returns
        ---------
        Tensor
        """
        return self._backend_distribution.sample(*args, **kwargs)

    def sample_numpy(self, *args, **kwargs):
        """Return raw numpy format of sampling"""
        return self.sample(*args, **kwargs).numpy()

    def log_prob(self, value):
        return self._backend_distribution.log_prob(value)
