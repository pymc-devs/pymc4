from pymc4.distributions.abstract.distribution import Distribution, Potential as BasePotential


class BackendDistribution(Distribution):
    """Backend distribution for Tensorflow distributions."""

    _backend_distribution = None

    def sample(self, shape=(), seed=None):
        return self._backend_distribution.sample(shape, seed)

    def sample_numpy(self, shape=(), seed=None):
        return self.sample(shape, seed).numpy()

    def log_prob(self, value):
        return self._backend_distribution.log_prob(value)

    def log_prob_numpy(self, value):
        return self.log_prob(value).numpy()


class Potential(BasePotential):
    def _init_backend(self):
        pass

    @property
    def value_numpy(self):
        return self.value.numpy()
