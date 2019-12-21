import types
from typing import Optional, Union, Tuple, List, Dict, Set, Any
import numpy as np
import tensorflow as tf
from pymc4.coroutine_model import Model
from pymc4.flow import evaluate_model, SamplingState, evaluate_model_posterior_predictive
from pymc4.inference.utils import trace_to_arviz
from pymc4.flow.executor import assert_values_compatible_with_distribution_shape


__all__ = ["sample_prior_predictive", "sample_posterior_predictive"]


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
    _, state = evaluate_model(model, state=state)
    distributions_names = list(state.untransformed_values)
    deterministic_names = list(state.deterministics)
    observed = None
    traced_observeds: Set[str] = set()
    if sample_from_observed:
        state.observed_values = observed = {k: None for k in state.observed_values}
        distributions_names = distributions_names + list(state.observed_values)

    if isinstance(var_names, str):
        var_names = [var_names]

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
    @tf.function(autograph=False)
    def single_draw(index):
        _, state = evaluate_model(model, observed=observed)
        return tuple(
            state.untransformed_values[k]
            if k in state.untransformed_values
            else (state.observed_values[k] if k in traced_observeds else state.deterministics[k])
            for k in var_names
        )

    # Make draws in parallel with tf.vectorized_map
    samples = tf.vectorized_map(single_draw, tf.range(int(np.prod(sample_shape))))

    # Convert the samples to ndarrays and make a dictionary with the desired sample_shape
    output = dict()
    for name, sample in zip(var_names, samples):
        sample = sample.numpy()
        output[name] = np.reshape(sample, sample_shape + sample.shape[1:])
    return trace_to_arviz(prior_predictive=output)


def sample_posterior_predictive(
    model: ModelType,
    trace: Dict[str, Any],
    var_names: Optional[List[str]] = None,
    observed: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Draw ``sample_shape`` values from the model for the desired ``var_names``.

    Parameters
    ----------
    model : types.GeneratorType, pymc4.Model
        Model to draw samples from
    trace: Dict[str, Any]
        The samples drawn from the model's posterior distribution that should
        be used for sampling from the posterior predictive
    var_names: Optional[List[str]]
        The list of variable names that will be included in the returned
        samples. If ``None``, the samples drawn for all observed
        distributions will be returned in the ``Samples`` dictionary.
    observed : Optional[Dict[str, Any]]
        A dictionary that can be used to override the distribution observed
        values defined in the model.

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
    ...     sd = yield pm.HalfNormal("sd", 5.)
    ...     norm = yield pm.Normal("n", 0, sd, observed=np.random.randn(100))

    Now, we may want to draw samples from the model's posterior to then sample
    from the posterior predictive.

    >>> trace, stats = pm.inference.sampling.sample(model())
    >>> ppc = pm.sample_posterior_predictive(model(), trace)

    The samples are returned as a dictionary with the variable names as keys

    >>> sorted(list(ppc))
    ['model/n']

    The drawn values are the dictionary's values, and their shape will depend
    on the supplied ``trace``

    >>> ppc["model/n"].shape
    (1000, 10, 100)

    """
    if var_names is not None and len(var_names) == 0:
        raise ValueError("Supplied an empty var_names list to sample from")

    if isinstance(var_names, str):
        var_names = [var_names]

    # We cannot assume that the model is vectorized, so we have batch the
    # pm.evaluate_model_posterior_predictive calls across the trace entries
    # This brings one big problem: we need to infer the batch dimensions from
    # the trace. To do this, we will do
    # 1) A regular single forward pass to determine the variable's shapes
    #    (we'll call these the core shapes)
    # 2) Go through the supplied trace to get each variable's batch shapes
    #    (the shapes to the left of the core shapes)
    # 3) Broadcast the encountered batch shapes between each other as a sanity
    #    check to get the global trace's batch_shape
    # 4) Broadcast the values in the trace to the global batch_shape to get
    #    each variable's broadcasted value.
    # 5) As tf.vectorized_map only iterates across the first dimension, we want
    #    to flatten the batch dimensions. To do this, we reshape the broadcasted
    #    values to (-1,) + core_shape. This way, tf.vectorized_map will be able
    #    to vectorize across the entire batch
    # 6) Collect the samples from, reshape them to batch_shape + core_shape and
    #    return them

    # Do a single forward pass to infer the distributions core shapes and
    # default observeds
    state = evaluate_model_posterior_predictive(model, observed=observed)[1]
    if var_names is None:
        var_names = list(state.posterior_predictives)
    else:
        defined_variables = set(state.all_values) | set(state.deterministics)
        if not set(var_names) <= defined_variables:
            raise KeyError(
                "The supplied var_names = {} are not defined in the model.\n"
                "Defined variables are = {}".format(
                    list(set(var_names) - defined_variables), list(defined_variables)
                )
            )

    # Get the global batch_shape
    batch_shape = tf.TensorShape([])
    # Get a copy of trace because we may manipulate the dictionary later in this
    # function
    posterior = trace.posterior.copy()
    posterior_names = list(posterior)
    for var_name in posterior_names:
        values = tf.convert_to_tensor(posterior[var_name].values)
        try:
            core_shape = state.all_values[var_name].shape
        except KeyError:
            if var_name in state.deterministics:
                # Remove the deterministics from the trace
                del trace[var_name]
                continue
            else:
                raise TypeError(
                    "Supplied the variable {} in the trace, yet this variable is "
                    "not defined in the model: {!r}".format(var_name, state)
                )
        assert_values_compatible_with_distribution_shape(var_name, values, core_shape)
        batch_shape = tf.TensorShape(
            tf.broadcast_static_shape(
                values.shape[: len(values.shape) - len(core_shape)], batch_shape
            )
        )

    # Flatten the batch axis
    flattened_posterior = []
    for k, v in posterior.items():
        core_shape = tf.TensorShape(state.all_values[k].shape)
        batched_val = tf.broadcast_to(v.values, batch_shape + core_shape)
        flattened_posterior.append(tf.reshape(batched_val, shape=[-1] + core_shape.as_list()))
    posterior_vars = list(posterior)
    # Setup the function that makes a single draw
    @tf.function(autograph=False)
    def single_draw(elems):
        values = dict(zip(posterior_vars, elems))
        _, st = evaluate_model_posterior_predictive(model, values=values, observed=observed)
        return tuple(
            [
                (
                    st.untransformed_values[k]
                    if k in st.untransformed_values
                    else (
                        st.deterministics[k] if k in st.deterministics else st.transformed_values[k]
                    )
                )
                for k in var_names
            ]
        )

    # Make draws in parallel across the batch elements with tf.vectorized_map
    samples = tf.vectorized_map(single_draw, flattened_posterior)
    # Convert the samples to ndarrays and make a dictionary with the correct
    # batch_shape + core_shape
    output = dict()
    for name, sample in zip(var_names, samples):
        sample = sample.numpy()
        output[name] = np.reshape(sample, batch_shape + sample.shape[1:])
    return trace_to_arviz(posterior_predictive=output)
