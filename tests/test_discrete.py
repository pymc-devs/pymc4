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
        disc = yield pm.Normal("disc", 0, 1)
        return disc

    return simple_model


@pytest.fixture(scope="function")
def model_with_discrete():
    @pm.model()
    def model_with_discrete():
        disc = yield pm.Categorical("disc", probs=[0.1, 0.9])
        return disc

    return model_with_discrete


@pytest.fixture(scope="function")
def model_with_discrete_and_continuous():
    @pm.model()
    def model_with_discrete_and_continuous():
        disc = yield pm.Categorical("disc", probs=[0.1, 0.9])
        norm = yield pn.Normal("mu", 0, 1)
        return norm

    return model_with_discrete_and_continuous


@pytest.fixture(scope="module", params=["XLA", "noXLA"], ids=str)
def xla_fixture(request):
    return request.param == "XLA"


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


def test_discrete_sampling(model_with_discrete, xla_fixture):
    model = model_with_discrete()
    with pytest.raises(Exception) as exinfo:
        trace = pm.sample(model=model, sample_type="randomwalk", xla_fixture=xla_fixture)


def test_discrete_sampling(model_with_discrete_and_continuous, xla_fixture):
    model = model_with_discrete()
    with pytest.raises(Exception) as exinfo:
        trace = pm.sample(model=model, sample_type="compound", xla_fixture=xla_fixture)

