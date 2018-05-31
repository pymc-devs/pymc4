from tensorflow_probability import edward2 as ed
from pymc4 import Model

__all__ = ["RandomVariable"]


class RandomVariable(ed.RandomVariable):

    def __init__(
                self,
                distribution,
                sample_shape=(),
                value=None,
                name="RV"
                ):
        self.model = Model.get_context()
        self.name = name

        super(RandomVariable, self).__init__(
                                                distribution,
                                                sample_shape,
                                                value,
                                                )

        self.model.add_random_variable(self)
