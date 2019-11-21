import types
from typing import Optional, Union, Tuple, List, Dict, Set
import numpy as np
import tensorflow as tf
from pymc4.coroutine_model import Model
from pymc4.flow import evaluate_model, SamplingState


__all__ = ["sample_prior_predictive"]


ModelType = Union[types.GeneratorType, Model]
MODEL_TYPES = (types.GeneratorType, Model)


def sample_prior_predictive(
    model: ModelType,
    sample_shape: Union[int, Tuple[int]] = 1000,
    sample_from_observed: bool = True,
    var_names: Optional[List[str]] = None,
    state: Optional[SamplingState] = None,
) -> Dict[str, np.ndarray]:
    """
    Draw ``sample_shape`` values from the model for the desired ``var_names``.

    Parameters
    ----------
    model : types.GeneratorType, pymc4.Model
        Model to draw samples from
    sample_shape: Union[int, Tuple[int]]
        The sample shape of the draw. Every distribution has its core
        dimensions (e.g. ``pm.Normal("x", 0, tf.ones(2))`` has a single core
        dimension with ``shape=(2,)``). The ``sample_shape`` controls the total
        number of draws to make from a distribution, and the shape that will
        be prepended to the core dimensions. In the above case, if
        ``sample_shape=(3, 1)``, then the resulting draw will have
        ``shape=(3, 1, 2)``. If an ``int`` is passed, it's converted to a tuple
        with a single entry: ``(sample_shape,)``
    sample_from_observed: bool
        If ``False``, the distributions that were assigned observed values wont
        be resampled, and the observed values will used for computations
        downstream.
        If ``True``, the distributions that were assigned observed values will
        be resampled. This means that their observed value will be completely
        ignored (including its implied shape), and a new sample will be drawn
        from the prior distribution.
        Observed variables are only returned in the ``Samples`` dictionary if
        ``sample_from_observed`` is ``True`` or the name of the observed
        variable is explicitly provided in ``var_names``.
    var_names: Optional[List[str]]
        The list of variable names that will be included in the returned
        samples. If ``None``, the samples drawn for all untransformed
        distributions and deterministics will be returned in the ``Samples``
        dictionary. Furthermore, if ``sample_from_observed=True``, then the
        observed variable names will be added to the untransformed
        distributions.
    state : Optional[pymc4.flow.SamplingState]
        A ``SamplingState`` that can be used to specify distributions fixed
        values and change observed values.

    Returns
    -------
    Samples: Dict[str, np.ndarray]
        A dictionary of ``var_names`` keys and their corresponding drawn
        samples.

    Examples
    --------
    Lets define a simple model to sample from

    >>> import pymc4 as pm
    >>> @pm.model
    ... def model():
    ...     sd = yield pm.HalfNormal("sd", 1.)
    ...     norm = yield pm.Normal("n", 0, sd, observed=np.random.randn(10))

    Now, we may want to draw samples from the model's prior, ignoring the
    observed values.

    >>> prior_samples = sample_prior_predictive(model(), sample_shape=(20, 3))

    The samples are returned as a dictionary with the variable names as keys

    >>> sorted(list(prior_samples))
    ['model/n', 'model/sd']

    The drawn values are the dictionary's values, and their shape will depend
    on the supplied ``sample_shape``

    >>> [v.shape for v in prior_samples.values()]
    [(20, 3), (20, 3)]

    If we only wanted to draw samples from unobserved variables we would
    have done the following

    >>> prior_samples = sample_prior_predictive(model(), sample_from_observed=False)
    >>> sorted(list(prior_samples))
    ['model/sd']

    Notes
    -----
    If ``sample_from_observed=False``, the observed value passed to the
    variables will be used in the later stages of the model's computation.

    >>> import pymc4 as pm
    >>> @pm.model
    ... def model2():
    ...     sd = yield pm.HalfNormal("sd", 1.)
    ...     x = yield pm.Normal("x", 0, sd, observed=np.ones(10))
    ...     y = yield pm.Normal("y", x, 1e-8)
    >>> prior_samples = sample_prior_predictive(
    ...     model2(), sample_shape=(20,), sample_from_observed=False
    ... )
    >>> np.allclose(np.mean(prior_samples["model2/y"]), 1)
    True

    Furthermore, this has consequences at the shape level of the drawn samples
    >>> prior_samples["model2/y"].shape
    (20, 10)

    If ``sample_from_observed=True`` the value of the ``x`` random variable
    will be drawn from its prior distribution, which will have consequences
    both at the value and shape levels of downstream computations

    >>> prior_samples = sample_prior_predictive(
    ...     model2(), sample_shape=(20,), sample_from_observed=True
    ... )
    >>> np.allclose(np.mean(prior_samples["model2/y"]), 1)
    False
    >>> prior_samples["model2/y"].shape
    (20,)

    """
    if isinstance(sample_shape, int):
        sample_shape = (sample_shape,)

    # Do a single forward pass to establish the distributions, deterministics and observeds
    state = evaluate_model(model, state=state)[1]
    distributions_names = list(state.untransformed_values)
    deterministic_names = list(state.deterministics)
    observed = None
    traced_observeds: Set[str] = set()
    if sample_from_observed:
        state.observed_values = observed = {k: None for k in state.observed_values}
        distributions_names = distributions_names + list(state.observed_values)
    if var_names is None:
        var_names = distributions_names + deterministic_names
    else:
        # We can trace the observed values if their names are explicitly requested in var_names
        traced_observeds = set(
            [var_name for var_name in var_names if var_name in state.observed_values]
        )
    if not set(var_names) <= (set(distributions_names + deterministic_names) | traced_observeds):
        raise ValueError(
            "Some of the supplied var_names are not defined in the supplied "
            "model {}.\nList of unknown var_names: {}".format(
                model, list(set(var_names) - set(distributions_names + deterministic_names))
            )
        )

    # Setup the function that makes a single draw
    def single_draw(index):
        _, st = evaluate_model(model, observed=observed)
        return tuple(
            [
                (
                    st.untransformed_values[k]
                    if k in st.untransformed_values
                    else (st.observed_values[k] if k in traced_observeds else st.deterministics[k])
                )
                for k in var_names
            ]
        )

    # Make draws in parallel with tf.vectorized_map
    samples = tf.vectorized_map(single_draw, tf.range(int(np.prod(sample_shape))))

    # Convert the samples to ndarrays and make a dictionary with the desired sample_shape
    output = dict()
    for name, sample in zip(var_names, samples):
        sample = sample.numpy()
        output[name] = np.reshape(sample, sample_shape + sample.shape[1:])
    return output
