from pymc4.distributions.abstract.distribution import Distribution, Potential as BasePotential
from tensorflow_probability import distributions as tfd


class BackendDistribution(Distribution):
    """Backend distribution for Tensorflow distributions."""

    _backend_distribution: tfd.Distribution = None  # make type checkers happy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.plate is not None:
            self._backend_distribution = tfd.Sample(
                self._backend_distribution, sample_shape=self.plate
            )

    def sample(self, shape=(), seed=None):
        return self._backend_distribution.sample(shape, seed)

    def sample_numpy(self, shape=(), seed=None):
        return self.sample(shape, seed).numpy()

    def log_prob(self, value):
        return self._backend_distribution.log_prob(value)

    def log_prob_numpy(self, value):
        return self.log_prob(value).numpy()


class Potential(BasePotential):
    @property
    def value_numpy(self):
        return self.value.numpy()
