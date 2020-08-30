"""PyMC4 test configuration."""
import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf
import itertools

# Tensor shapes on which the GP model will be tested
BATCH_AND_FEATURE_SHAPES = [
    (1,),
    (2,),
    (2, 2,),
]
SAMPLE_SHAPE = [(1,), (3,)]


@pytest.fixture(scope="function", autouse=True)
def tf_seed():
    tf.random.set_seed(37208)  # random.org
    yield


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
        norm = yield pm.Normal("norm", 0, 1, batch_stack=norm_shape)
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
        "outer_model/nested_model/dx": (["outer_model/nested_model/x"], lambda x: x + 1,),
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
                event_stack=5,
            )

    else:

        @pm.model
        def model():
            mu = yield pm.Normal("mu", tf.zeros(4), 1)
            scale = yield pm.HalfNormal("scale", 1)
            x = yield pm.Normal("x", mu, scale, batch_stack=5, observed=observed)

    return model, is_vectorized_model, core_shapes


@pytest.fixture(scope="module", params=BATCH_AND_FEATURE_SHAPES, ids=str)
def get_batch_shape(request):
    return request.param


@pytest.fixture(scope="module", params=SAMPLE_SHAPE, ids=str)
def get_sample_shape(request):
    return request.param


@pytest.fixture(scope="module", params=BATCH_AND_FEATURE_SHAPES, ids=str)
def get_feature_shape(request):
    return request.param


@pytest.fixture(scope="module")
def get_data(get_batch_shape, get_sample_shape, get_feature_shape):
    X = tf.random.normal(get_batch_shape + get_sample_shape + get_feature_shape)
    return get_batch_shape, get_sample_shape, get_feature_shape, X
