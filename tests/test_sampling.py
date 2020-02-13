import pytest
import itertools
import pymc4 as pm
import numpy as np
from scipy import stats
import tensorflow as tf


@pytest.fixture(scope="function")
def simple_model():
    @pm.model()
    def simple_model():
        norm = yield pm.Normal("norm", 0, 1)
        return norm

    return simple_model


@pytest.fixture(scope="function")
def simple_model_with_deterministic(simple_model):
    @pm.model()
    def simple_model_with_deterministic():
        norm = yield simple_model()
        determ = yield pm.Deterministic("determ", norm * 2)
        return determ

    return simple_model_with_deterministic


@pytest.fixture(scope="function")
def simple_model_no_free_rvs():
    @pm.model()
    def simple_model_no_free_rvs():
        norm = yield pm.Normal("norm", 0, 1, observed=1)
        return norm

    return simple_model_no_free_rvs


@pytest.fixture(
    scope="function",
    params=itertools.product(
        [(), (3,), (3, 2)], [(), (2,), (4,), (5, 4)], [(), (1,), (10,), (10, 10)]
    ),
    ids=str,
)
def unvectorized_model(request):
    norm_shape, observed_shape, batch_size = request.param
    observed = np.ones(observed_shape)

    @pm.model()
    def unvectorized_model():
        norm = yield pm.Normal("norm", 0, 1, plate=norm_shape)
        determ = yield pm.Deterministic("determ", tf.reduce_max(norm))
        output = yield pm.Normal("output", determ, 1, observed=observed)

    return unvectorized_model, norm_shape, observed, batch_size


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


@pytest.fixture(scope="function")
def deterministics_in_nested_models():
    @pm.model
    def nested_model(cond):
        x = yield pm.Normal("x", cond, 1)
        dx = yield pm.Deterministic("dx", x + 1)
        return dx

    @pm.model
    def outer_model():
        cond = yield pm.HalfNormal("cond", 1)
        dcond = yield pm.Deterministic("dcond", cond * 2)
        dx = yield nested_model(dcond)
        ddx = yield pm.Deterministic("ddx", dx)
        return ddx

    expected_untransformed = {
        "outer_model",
        "outer_model/cond",
        "outer_model/nested_model",
        "outer_model/nested_model/x",
    }
    expected_transformed = {"outer_model/__log_cond"}
    expected_deterministics = {
        "outer_model/dcond",
        "outer_model/ddx",
        "outer_model/nested_model/dx",
    }
    deterministic_mapping = {
        "outer_model/dcond": (["outer_model/__log_cond"], lambda x: np.exp(x) * 2),
        "outer_model/ddx": (["outer_model/nested_model/dx"], lambda x: x),
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1),
    }

    return (
        outer_model,
        expected_untransformed,
        expected_transformed,
        expected_deterministics,
        deterministic_mapping,
    )


@pytest.fixture(scope="module", params=["auto_batch", "trust_manual_batching"], ids=str)
def use_auto_batching_fixture(request):
    return request.param == "auto_batch"


@pytest.fixture(scope="function", params=["unvectorized_model", "vectorized_model"], ids=str)
def vectorized_model_fixture(request):
    is_vectorized_model = request.param == "vectorized_model"
    observed = np.zeros((5, 4), dtype="float32")
    core_shapes = {
        "model/mu": (4,),
        "model/__log_scale": (),
    }
    if is_vectorized_model:
        # A model where we pay great attention to making each distribution
        # have exactly the right event_shape, and assure that when we sample
        # from its prior, the requested `sample_shape` gets sent to the
        # conditionally independent variables, and expect that shape to go
        # through the conditionally dependent variables as batch_shapes
        @pm.model
        def model():
            mu = yield pm.Normal(
                "mu", tf.zeros(4), 1, conditionally_independent=True, reinterpreted_batch_ndims=1,
            )
            scale = yield pm.HalfNormal("scale", 1, conditionally_independent=True)
            x = yield pm.Normal(
                "x",
                mu,
                scale[..., None],
                observed=observed,
                reinterpreted_batch_ndims=1,
                plate_events=5,
            )

    else:

        @pm.model
        def model():
            mu = yield pm.Normal("mu", tf.zeros(4), 1)
            scale = yield pm.HalfNormal("scale", 1)
            x = yield pm.Normal("x", mu, scale, plate=5, observed=observed)

    return model, is_vectorized_model, core_shapes


def test_sample_deterministics(simple_model_with_deterministic, xla_fixture):
    model = simple_model_with_deterministic()
    trace = pm.sample(
        model=model, num_samples=10, num_chains=4, burn_in=100, step_size=0.1, xla=xla_fixture
    )
    norm = "simple_model_with_deterministic/simple_model/norm"
    determ = "simple_model_with_deterministic/determ"
    np.testing.assert_allclose(trace.posterior[determ], trace.posterior[norm] * 2)


def test_vectorize_log_prob_det_function(unvectorized_model):
    model, norm_shape, observed, batch_size = unvectorized_model
    model = model()
    (
        logpfn,
        all_unobserved_values,
        deterministics_callback,
        deterministic_names,
        state,
    ) = pm.inference.sampling.build_logp_and_deterministic_functions(model)
    for _ in range(len(batch_size)):
        logpfn = pm.inference.sampling.vectorize_logp_function(logpfn)
        deterministics_callback = pm.inference.sampling.vectorize_logp_function(
            deterministics_callback
        )

    # Test function inputs and initial values are as expected
    assert set(all_unobserved_values) <= {"unvectorized_model/norm"}
    assert all_unobserved_values["unvectorized_model/norm"].numpy().shape == norm_shape
    assert set(deterministic_names) <= {"unvectorized_model/determ"}

    # Setup inputs to vectorized functions
    inputs = np.random.normal(size=batch_size + norm_shape).astype("float32")
    input_tensor = tf.convert_to_tensor(inputs)

    # Test deterministic part
    expected_deterministic = np.max(np.reshape(inputs, batch_size + (-1,)), axis=-1)
    deterministics_callback_output = deterministics_callback(input_tensor)[0].numpy()
    assert deterministics_callback_output.shape == batch_size
    np.testing.assert_allclose(deterministics_callback_output, expected_deterministic, rtol=1e-5)

    # Test log_prob part
    expected_log_prob = np.sum(
        np.reshape(stats.norm.logpdf(inputs), batch_size + (-1,)), axis=-1
    ) + np.sum(  # norm.log_prob
        stats.norm.logpdf(observed.flatten(), loc=expected_deterministic[..., None], scale=1),
        axis=-1,
    )  # output.log_prob
    logpfn_output = logpfn(input_tensor).numpy()
    assert logpfn_output.shape == batch_size
    np.testing.assert_allclose(logpfn_output, expected_log_prob, rtol=1e-5)


def test_sampling_with_deterministics_in_nested_models(
    deterministics_in_nested_models, xla_fixture
):
    (
        model,
        expected_untransformed,
        expected_transformed,
        expected_deterministics,
        deterministic_mapping,
    ) = deterministics_in_nested_models
    trace = pm.sample(
        model=model(), num_samples=10, num_chains=4, burn_in=100, step_size=0.1, xla=xla_fixture
    )
    for deterministic, (inputs, op) in deterministic_mapping.items():
        np.testing.assert_allclose(
            trace.posterior[deterministic], op(*[trace.posterior[i] for i in inputs]), rtol=1e-6
        )


def test_sampling_with_no_free_rvs(simple_model_no_free_rvs):
    model = simple_model_no_free_rvs()
    with pytest.raises(ValueError):
        trace = pm.sample(model=model, num_samples=1, num_chains=1, burn_in=1)


def test_sample_auto_batching(vectorized_model_fixture, xla_fixture, use_auto_batching_fixture):
    model, is_vectorized_model, core_shapes = vectorized_model_fixture
    num_samples = 10
    num_chains = 4
    if not is_vectorized_model and not use_auto_batching_fixture:
        with pytest.raises(Exception):
            pm.inference.sampling.sample(
                model=model(),
                num_samples=num_samples,
                num_chains=num_chains,
                burn_in=1,
                step_size=0.1,
                xla=xla_fixture,
                use_auto_batching=use_auto_batching_fixture,
            )
    else:
        trace = pm.inference.sampling.sample(
            model=model(),
            num_samples=num_samples,
            num_chains=num_chains,
            burn_in=1,
            step_size=0.1,
            xla=xla_fixture,
            use_auto_batching=use_auto_batching_fixture,
        )
        posterior = trace.posterior
        for rv_name, core_shape in core_shapes.items():
            assert posterior[rv_name].shape == (num_chains, num_samples) + core_shape


def test_beta_sample():
    @pm.model
    def model():
        dist = yield pm.Beta("beta", 0, 1)
        return dist

    trace = pm.sample(model(), num_samples=1, burn_in=1)

    assert trace.posterior["model/beta"] is not None
    assert trace.posterior["model/__sigmoid_beta"] is not None
