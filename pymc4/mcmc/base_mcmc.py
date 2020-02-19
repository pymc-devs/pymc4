from pymc4.inference.utils import initialize_state
from pymc4.coroutine_model import Model

__all__ = ["SamplerConstr"]


class SamplerConstr:
    def __new__(cls, *args, **kwargs):
        if not args:
            raise ValueError(
                "Sampler class should be provided with `pymc4.Model` object as the first argument"
            )
        model = args[0]
        if not isinstance(model, Model):
            raise TypeError(
                "`sample` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                    type(model)
                )
            )
        non_sampling_state = initialize_state(model)
        discrete_names, continuous_names = (
            list(non_sampling_state.discrete_distributions),
            list(non_sampling_state.continuous_distributions),
        )
        if cls._grad is True and discrete_names:
            raise ValueError("Discrete distributions can't be used with gradient-based sampler")
        # TODO: add Compound support in class constructor?
        return super().__new__(cls)
