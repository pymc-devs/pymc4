from ..abstract.distribution import Distribution


class BackendDistribution(Distribution):
    """
    BackendDistribution for Tensorflow distributions
    """
    __doc__ = Distribution.__doc__ + __doc__

    def sample(self, shape=(), seed=None):
        return self._backend_distribution.sample(shape, seed)

    def sample_numpy(self, shape=(), seed=None):
        return self.sample(shape, seed).numpy()

    def log_prob(self, value):
        return self._backend_distribution.log_prob(value)

    def log_prob_numpy(self, value):
        return self.log_prob(value).numpy()
