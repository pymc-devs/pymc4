"""
Fixtures for test_sampling and test_forward_sampling
"""
import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="module", params=[(1, 0), (1, 1), (1, 2), (1, 1, 1), (1, 3, 7)], ids=str)
def sample_shape(request):
    return request.param


# Previously called vectorized_model_fixture in conftest.py and test_forward_sampling.py
# Combined into one which required moving out "core_shapes" - see below
@pytest.fixture(scope="function", params=["unvectorized_model", "vectorized_model"], ids=str)
def vectorized_model_fixture(request):
    is_vectorized_model = request.param == "vectorized_model"
    observed = np.zeros((5, 4), dtype="float32")
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
                event_stack=5,
            )

    else:

        @pm.model
        def model():
            mu = yield pm.Normal("mu", tf.zeros(4), 1)
            scale = yield pm.HalfNormal("scale", 1)
            x = yield pm.Normal("x", mu, scale, batch_stack=5, observed=observed)

    return model, is_vectorized_model


@pytest.fixture(scope="module", params=["NoResampleObserved", "ResampleObserveds"], ids=str)
def sample_from_observed(request):
    return request.param == "ResampleObserveds"


# TODO
# This is the only difference between the sampling / forward sampling model in the vectorized models
# so moving this out until I understand the difference in the two use cases and how to refactor to suit
@pytest.fixture(scope="function")
def sampling_core_shapes():
    core_shapes = {
        "model/mu": (4,),
        "model/__log_scale": (),
    }
    return core_shapes


@pytest.fixture(scope="function")
def forward_sampling_core_shapes():
    core_shapes = {
        "model/mu": (4,),
        "model/scale": (),  # for forward sampling only
        "model/x": (5, 4),  # for forward sampling only
    }
    return core_shapes


# Previously called model_fixture in test_forward_sampling.py
# Suggest could be called something better?
@pytest.fixture(scope="function")
def model_fixture():
    observed = np.random.randn(10).astype("float32") + 1

    @pm.model
    def model():
        sd = yield pm.HalfNormal("sd", 1.0)
        mu = yield pm.Deterministic("mu", tf.convert_to_tensor(1.0))
        x = yield pm.Normal("x", mu, sd, observed=observed)
        y = yield pm.Normal("y", x, 1e-9)
        dy = yield pm.Deterministic("dy", 2 * y)

    return model, observed


# Previously called model_with_observed_fixture in test_forward_sampling.py
# Suggest could be called something better?
@pytest.fixture(scope="module", params=["observed_in_RV", "observed_in_eval"])
def model_with_observed(request):
    observed_in_RV = request.param == "observed_in_RV"
    observed = {
        "model/x": np.ones(10, dtype="float32"),
        "model/y": np.ones(1, dtype="float32"),
        "model/z": np.ones((10, 10), dtype="float32"),
    }
    observed_kwargs = {k: v if observed_in_RV else None for k, v in observed.items()}
    core_ppc_shapes = {
        "model/sd": (),
        "model/x": (10,),
        "model/y": (1,),
        "model/d": (10,),
        "model/z": (10, 10),
        "model/u": (10, 10),
    }

    @pm.model
    def model():
        sd = yield pm.Exponential("sd", 1)
        x = yield pm.Normal("x", 0, 1, observed=observed_kwargs["model/x"])
        y = yield pm.HalfNormal("y", 1, observed=observed_kwargs["model/y"])
        d = yield pm.Deterministic("d", x + y)
        z = yield pm.Normal("z", d, sd, observed=observed_kwargs["model/z"])
        u = yield pm.Exponential("u", z)
        return u

    return model, observed, core_ppc_shapes, observed_in_RV


# Previously called posterior_predictive_fixture in test_forward_sampling.py
@pytest.fixture(scope="module")
def posterior_predictive(model_with_observed):
    num_samples = 40
    num_chains = 3
    (model, observed, core_ppc_shapes, observed_in_RV) = model_with_observed
    trace = pm.sample(model(), num_samples=num_samples, num_chains=num_chains, observed=observed)
    return model, observed, core_ppc_shapes, observed_in_RV, trace, num_samples, num_chains


# Previously called glm_model_fixture in test_forward_sampling.py
@pytest.fixture(scope="module", params=["unvectorized_model", "vectorized_model"], ids=str)
def glm_model(request):
    is_vectorized_model = request.param == "vectorized_model"
    n_features = 10
    n_observations = 5
    regressors = np.zeros((n_observations, n_features), dtype="float32")
    observed = np.zeros((n_observations,), dtype="float32")
    core_shapes = {
        "model/beta": (n_features,),
        "model/bias": (),
        "model/scale": (),
        "model/y": (n_observations,),
    }
    if is_vectorized_model:

        @pm.model
        def model():
            beta = yield pm.Normal(
                "beta",
                tf.zeros((n_features,)),
                1,
                conditionally_independent=True,
                reinterpreted_batch_ndims=1,
            )
            bias = yield pm.Normal("bias", 0, 1, conditionally_independent=True)
            scale = yield pm.HalfNormal("scale", 1, conditionally_independent=True)
            mu = tf.linalg.matvec(regressors, beta) + bias[..., None]
            y = yield pm.Normal(
                "y", mu, scale[..., None], observed=observed, reinterpreted_batch_ndims=1,
            )

    else:

        @pm.model
        def model():
            beta = yield pm.Normal("beta", tf.zeros((n_features,)), 1)
            bias = yield pm.Normal("bias", 0, 1)
            scale = yield pm.HalfNormal("scale", 1)
            mu = tf.linalg.matvec(regressors, beta) + bias
            y = yield pm.Normal("y", mu, scale, observed=observed)

    return model, is_vectorized_model, core_shapes
