class RandomVariable:
    def __init__(self):
        self._parametrization = None

    def sample(self):
        pass

    def as_tensor(self):
        pass

    def make_parametrization(self):
        pass


class Parametrization:
    def __init__(self):
        self._free_vars = []
        self._trainable_params = []
        self._observation = None

    def log_prob(self, value, **vars):
        pass

    def forward(self, **free_vars):
        pass

    def backward(self, observed):
        return NotImplemented


class DirectParametrization:
    def __init__(self, dist):
        self._dist = dist
        self._free_vars = []
        self._trainable_params = []

    def log_prob(self, observed):
        return self._dist.log_prob(observed)


class TransformedParametrization:
    def __init__(self, dist, bijector):
        self._bijector = bijector
        self._dist = dist

    def log_prob(self, observed):
        raise NotImplementedError
