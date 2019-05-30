from ..coroutine_model import Model
from ..scopes import Scope


class Distribution(Model):
    def __init__(self, name, keep_auxiliary=False, keep_return=True, transform=None):
        super().__init__(self.control_flow, name=name, keep_return=keep_return, keep_auxiliary=keep_auxiliary)
        self.transform = transform

    def sample(self, shape=(), seed=None):
        """
        Forward sampling implementation

        Parameters
        ----------
        shape : tuple
            sample shape
        seed : int|None
            random seed
        """
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def control_flow(self):
        if self.transform is not None and Scope.use_transform():
            value = yield from self.transformed_control_flow()
        else:
            value = yield from self.untransformed_control_flow()
        return value

    def transformed_control_flow(self):
        raise NotImplementedError

    def untransformed_control_flow(self):
        value = yield self
        return value


DIST_CONVERTERS = {
    # this should be ready to use
    Distribution: lambda x: x
}


def convert_distribution(dist):
    matched = list(filter(lambda key: isinstance(dist, key), DIST_CONVERTERS))
    if matched:
        return DIST_CONVERTERS[matched[-1]](dist)
    else:
        raise TypeError("object {} can't be converted to a PyMC4 distribution".format(dist))


def register_distribution_converter_function(cls):
    def wrap(func):
        DIST_CONVERTERS[cls] = func
        return func
    return wrap


