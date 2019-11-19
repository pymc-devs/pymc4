import pytest
import itertools
import pymc4 as pm
import numpy as np
from scipy import stats
from pymc4 import distributions as dist
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
        return output

    return unvectorized_model, norm_shape, observed, batch_size


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


def test_sample_deterministics(simple_model_with_deterministic, xla_fixture):
    if xla_fixture:
        pytest.skip("XLA in sampling is still not fully supported")
    model = simple_model_with_deterministic()
    trace, stats = pm.inference.sampling.sample(
        model=model, num_samples=10, num_chains=4, burn_in=100, step_size=0.1, xla=xla_fixture
    )
    norm = "simple_model_with_deterministic/simple_model/norm"
    determ = "simple_model_with_deterministic/determ"
    np.testing.assert_allclose(trace[determ], trace[norm] * 2)


def test_vectorize_log_prob_det_function(unvectorized_model):
    model, norm_shape, observed, batch_size = unvectorized_model
    model = model()
    (
        logpfn,
        all_unobserved_values,
        deterministics_callback,
        deterministic_names,
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
