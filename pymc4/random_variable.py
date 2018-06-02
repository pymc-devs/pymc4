from tensorflow_probability import edward2 as ed
from pymc4 import Model

__all__ = ["RandomVariable"]


class RandomVariable(ed.RandomVariable):

    def __init__(
            self,
            name,
            distribution,
            sample_shape=(),
            value=None,
    ):
        self.context_stack = Model.get_contexts()
        self.model = Model.get_context()
        if self.model.name is not "":
            self.name = self.model.name + "_" + name
        else:
            self.name = name

        super(RandomVariable, self).__init__(
            distribution,
            sample_shape,
            value,
        )

        for model in self.context_stack:
            model.add_random_variable(self)
