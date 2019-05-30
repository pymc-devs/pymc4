from ..base import Distribution as BaseDistribution, register_distribution_converter_function
from pymc4 import name_scope
from tensorflow_probability import bijectors as bij
from tensorflow_probability import distributions as tfd


class TensorflowDistributionAdapter(BaseDistribution):
    def __init__(self, name, dist: tfd.Distribution, *, keep_auxiliary=False, keep_return=True, transform=None):
        super().__init__(name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return, transform=transform)
        self.dist = dist

    def transformed_control_flow(self):
        inverse_transform = bij.Invert(self.transform)
        new_dist = tfd.TransformedDistribution(self.dist, inverse_transform)
        with name_scope("transformed"):
            new_dist = TensorflowDistributionAdapter("auxiliary", new_dist)
            unconstrained = yield new_dist
        return self.transform(unconstrained)

    def sample(self, shape=(), seed=None, **kwargs):
        return self.dist.sample(shape, seed=seed, **kwargs)

    def log_prob(self, value, **kwargs):
        return self.dist.log_prob(value, **kwargs)


@register_distribution_converter_function(tfd.Distribution)
def convert_tensorflow_distribution(dist: tfd.Distribution):
    name = dist.parameters["name"]
    return TensorflowDistributionAdapter(name, dist)
