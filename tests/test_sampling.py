import pytest
import pymc4 as pm
import numpy as np
from scipy import stats
import tensorflow as tf


def test_sample_deterministics(simple_model_with_deterministic, xla_fixture):
    model = simple_model_with_deterministic()
    trace = pm.sample(
        model=model,
        num_samples=10,
        num_chains=4,
        burn_in=100,
        step_size=0.1,
        xla=xla_fixture,
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
    ) = pm.mcmc.samplers.build_logp_and_deterministic_functions(model)
    for _ in range(len(batch_size)):
        logpfn = pm.mcmc.samplers.vectorize_logp_function(logpfn)
        deterministics_callback = pm.mcmc.samplers.vectorize_logp_function(deterministics_callback)

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
        model=model(),
        num_samples=10,
        num_chains=4,
        burn_in=100,
        step_size=0.1,
        xla=xla_fixture,
    )
    for deterministic, (inputs, op) in deterministic_mapping.items():
        np.testing.assert_allclose(
            trace.posterior[deterministic],
            op(*[trace.posterior[i] for i in inputs]),
            rtol=1e-6,
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
            pm.sample(
                model=model(),
                num_samples=num_samples,
                num_chains=num_chains,
                burn_in=1,
                step_size=0.1,
                xla=xla_fixture,
                use_auto_batching=use_auto_batching_fixture,
            )
    else:
        trace = pm.sample(
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


def test_sampling_unknown_sampler(simple_model):
    model = simple_model()
    with pytest.raises(KeyError):
        trace = pm.sample(model=model, sampler_type="unknown")


def test_sampling_log_likelihood(vectorized_model_fixture):
    model, is_vectorized_model, core_shapes = vectorized_model_fixture
    num_samples = 10
    num_chains = 4
    trace = pm.sample(
        model=model(),
        num_samples=num_samples,
        num_chains=num_chains,
        burn_in=1,
        step_size=0.1,
        include_log_likelihood=True,
    )

    if is_vectorized_model:
        # only one log likeliood matrix
        assert trace.log_likelihood["model/x"].shape == (num_chains, num_samples)

    else:
        state, _ = pm.initialize_sampling_state(model())
        assert trace.log_likelihood["model/x"].shape == (
            num_chains,
            num_samples,
            *state.observed_values["model/x"].shape,
        )
