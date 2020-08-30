import pytest
import re
import collections
import numpy as np
import tensorflow as tf
import pymc4 as pm
from pymc4 import forward_sampling
from pymc4.flow.executor import EvaluationError


@pytest.fixture(scope="module", params=[(1, 0), (1, 1), (1, 2), (1, 1, 1), (1, 3, 7)], ids=str)
def sample_shape_fixture(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 10, 100], ids=str)
def n_draws_fixture(request):
    return request.param


@pytest.fixture(scope="module", params=["NoResampleObserved", "ResampleObserveds"], ids=str)
def sample_from_observed_fixture(request):
    return request.param == "ResampleObserveds"


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


@pytest.fixture(scope="module", params=["observed_in_RV", "observed_in_eval"])
def model_with_observed_fixture(request):
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


@pytest.fixture(scope="module", params=["auto_batch", "trust_manual_batching"], ids=str)
def use_auto_batching_fixture(request):
    return request.param == "auto_batch"


@pytest.fixture(scope="module", params=["unvectorized_model", "vectorized_model"], ids=str)
def vectorized_model_fixture(request):
    is_vectorized_model = request.param == "vectorized_model"
    observed = np.zeros((5, 4), dtype="float32")
    core_shapes = {
        "model/mu": (4,),
        "model/scale": (),
        "model/x": (5, 4),
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


@pytest.fixture(scope="module")
def posterior_predictive_fixture(model_with_observed_fixture):
    num_samples = 40
    num_chains = 3
    (model, observed, core_ppc_shapes, observed_in_RV) = model_with_observed_fixture
    trace = pm.sample(model(), num_samples=num_samples, num_chains=num_chains, observed=observed)
    return (
        model,
        observed,
        core_ppc_shapes,
        observed_in_RV,
        trace,
        num_samples,
        num_chains,
    )


@pytest.fixture(scope="module", params=["unvectorized_model", "vectorized_model"], ids=str)
def glm_model_fixture(request):
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


def test_sample_prior_predictive(model_fixture, sample_shape_fixture, sample_from_observed_fixture):
    model, observed = model_fixture

    prior = pm.sample_prior_predictive(
        model(), sample_shape_fixture[1:], sample_from_observed_fixture
    ).prior_predictive
    if sample_from_observed_fixture:
        assert set(["model/sd", "model/x", "model/y", "model/mu", "model/dy"]) == set(
            prior.data_vars
        )
        assert all((value.shape == sample_shape_fixture for value in prior.values()))
    else:
        assert set(["model/sd", "model/y", "model/mu", "model/dy"]) == set(prior.data_vars)
        assert all(
            (
                value.shape == sample_shape_fixture
                for name, value in prior.items()
                if name in {"model/sd", "model/mu"}
            )
        )
        assert all(
            (
                value.shape == sample_shape_fixture + observed.shape
                for name, value in prior.items()
                if name in {"model/x", "model/y", "model/dy"}
            )
        )
        assert np.allclose(prior["model/y"], observed, rtol=1e-5)
        assert np.allclose(prior["model/y"] * 2, prior["model/dy"])


def test_sample_prior_predictive_var_names(model_fixture):
    model, observed = model_fixture

    prior = pm.sample_prior_predictive(
        model(), var_names=["model/sd"], sample_shape=()
    ).prior_predictive
    assert set(prior) == set(["model/sd"])

    prior = pm.sample_prior_predictive(
        model(), var_names=["model/x", "model/y"], sample_shape=()
    ).prior_predictive
    assert set(prior) == set(["model/x", "model/y"])

    # Assert we can get the values of observeds if we ask for them explicitly
    # even if sample_from_observed_fixture is False
    prior = pm.sample_prior_predictive(
        model(), var_names=["model/x", "model/y"], sample_shape=(), sample_from_observed=False,
    ).prior_predictive
    assert set(prior) == set(["model/x", "model/y"])
    assert np.all(prior["model/x"] == observed)

    # Assert an exception is raised if we pass wrong names
    model_func = model()
    expected_message = "Some of the supplied var_names are not defined in the supplied model {}.\nList of unknown var_names: {}".format(
        model_func, ["X"]
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        prior = pm.sample_prior_predictive(model_func, var_names=["X", "model/y"], sample_shape=())


def test_sample_prior_predictive_int_sample_shape(model_fixture, n_draws_fixture):
    model, observed = model_fixture

    prior_int = forward_sampling.sample_prior_predictive(model(), sample_shape=n_draws_fixture)

    prior_tuple = forward_sampling.sample_prior_predictive(model(), sample_shape=(n_draws_fixture,))

    assert all(
        (
            prior_int.prior_predictive[k].shape == v.shape
            for k, v in prior_tuple.prior_predictive.items()
        )
    )


def test_posterior_predictive_executor(model_with_observed_fixture):
    model, observed, core_ppc_shapes, _ = model_with_observed_fixture
    _, prior_state = pm.evaluate_model_transformed(model(), observed=observed)
    _, ppc_state = pm.evaluate_model_posterior_predictive(model(), observed=observed)

    # Assert that a normal evaluation has all observeds and the values match
    # to the observations
    assert len(prior_state.observed_values) == 3
    for var, val in observed.items():
        assert np.all(prior_state.all_values[var] == val)

    # Assert that a posterior predictive evaluation has no observed values
    # but the shapes of the samples match the supplied observed shapes
    assert len(ppc_state.observed_values) == 0
    for var, shape in core_ppc_shapes.items():
        assert (
            collections.ChainMap(ppc_state.all_values, ppc_state.deterministics_values)[var]
            .numpy()
            .shape
            == shape
        )
        if var in observed:
            assert np.any(ppc_state.all_values[var] != val for val in observed.values())


def test_sample_posterior_predictive(posterior_predictive_fixture):
    (
        model,
        observed,
        core_ppc_shapes,
        observed_in_RV,
        trace,
        num_samples,
        num_chains,
    ) = posterior_predictive_fixture

    if observed_in_RV:
        observed_kwarg = None
    else:
        observed_kwarg = observed
    ppc = pm.sample_posterior_predictive(
        model(), trace, observed=observed_kwarg
    ).posterior_predictive
    assert set(sorted(list(ppc))) == set(observed)
    assert np.all(
        [v.shape == (num_chains, num_samples) + observed[k].shape for k, v in ppc.items()]
    )


def test_sample_ppc_var_names(model_fixture):
    model, observed = model_fixture
    trace = pm.mcmc.utils.trace_to_arviz(
        {
            "model/sd": tf.ones((10, 1), dtype="float32"),
            "model/y": tf.convert_to_tensor(observed[:, None]),
        }
    )

    with pytest.raises(ValueError):
        forward_sampling.sample_posterior_predictive(model(), trace, var_names=[])

    with pytest.raises(KeyError):
        forward_sampling.sample_posterior_predictive(
            model(), trace, var_names=["name not in model!"]
        )

    with pytest.raises(TypeError):
        trace.posterior["name not in model!"] = tf.constant(1.0)
        pm.sample_posterior_predictive(model(), trace)
    del trace.posterior["name not in model!"]

    var_names = ["model/sd", "model/x", "model/dy"]
    ppc = pm.sample_posterior_predictive(model(), trace, var_names=var_names).posterior_predictive
    assert set(var_names) == set(ppc)
    assert ppc["model/sd"].shape == trace.posterior["model/sd"].shape


def test_sample_ppc_corrupt_trace():
    @pm.model
    def model():
        x = yield pm.Normal("x", tf.ones(5), 1, reinterpreted_batch_ndims=1)
        y = yield pm.Normal("y", x, 1)

    trace1 = pm.mcmc.utils.trace_to_arviz({"model/x": tf.ones((7, 1), dtype="float32")})

    trace2 = pm.mcmc.utils.trace_to_arviz(
        {"model/x": tf.ones((1, 5), dtype="float32"), "model/y": tf.zeros((1, 1), dtype="float32"),}
    )
    with pytest.raises(EvaluationError):
        forward_sampling.sample_posterior_predictive(model(), trace1)
    with pytest.raises(EvaluationError):
        forward_sampling.sample_posterior_predictive(model(), trace2)


def test_vectorized_sample_prior_predictive(
    vectorized_model_fixture, use_auto_batching_fixture, sample_shape_fixture
):
    model, is_vectorized_model, core_shapes = vectorized_model_fixture
    prior = forward_sampling.sample_prior_predictive(
        model(), sample_shape=sample_shape_fixture, use_auto_batching=use_auto_batching_fixture,
    ).prior_predictive
    if not use_auto_batching_fixture and not is_vectorized_model and len(sample_shape_fixture) > 0:
        with pytest.raises(AssertionError):
            for k, v in core_shapes.items():
                # The (1,) comes from trace_to_arviz imposed chain axis
                assert prior[k].shape == (1,) + sample_shape_fixture + v
    else:
        for k, v in core_shapes.items():
            # The (1,) comes from trace_to_arviz imposed chain axis
            assert prior[k].shape == (1,) + sample_shape_fixture + v


def test_sample_prior_predictive_on_glm(
    glm_model_fixture, use_auto_batching_fixture, sample_shape_fixture
):
    model, is_vectorized_model, core_shapes = glm_model_fixture
    if not use_auto_batching_fixture and not is_vectorized_model and len(sample_shape_fixture) > 0:
        with pytest.raises(AssertionError):
            prior = forward_sampling.sample_prior_predictive(
                model(),
                sample_shape=sample_shape_fixture,
                use_auto_batching=use_auto_batching_fixture,
            ).prior_predictive
            for k, v in core_shapes.items():
                # The (1,) comes from trace_to_arviz imposed chain axis
                assert prior[k].shape == (1,) + sample_shape_fixture + v
    else:
        prior = forward_sampling.sample_prior_predictive(
            model(), sample_shape=sample_shape_fixture, use_auto_batching=use_auto_batching_fixture,
        ).prior_predictive
        for k, v in core_shapes.items():
            # The (1,) comes from trace_to_arviz imposed chain axis
            assert prior[k].shape == (1,) + sample_shape_fixture + v


def test_vectorized_sample_posterior_predictive(
    vectorized_model_fixture, use_auto_batching_fixture, sample_shape_fixture
):
    model, is_vectorized_model, core_shapes = vectorized_model_fixture
    trace = pm.mcmc.utils.trace_to_arviz(
        {
            # The transposition of the first two axis comes from trace_to_arviz
            # that does this to the output of `sample` to get (num_chains, num_samples, ...)
            # instead of (num_samples, num_chains, ...)
            k: tf.zeros(
                (sample_shape_fixture[1], sample_shape_fixture[0]) + sample_shape_fixture[2:] + v
            )
            for k, v in core_shapes.items()
            if k not in ["model/x"]
        }
    )
    if not use_auto_batching_fixture and not is_vectorized_model and len(sample_shape_fixture) > 0:
        with pytest.raises((ValueError, EvaluationError)):
            # This can raise ValueError when tfp distributions complain about
            # the parameter shapes being imcompatible or it can raise an
            # EvaluationError because the distribution shape is not compatible
            # with the supplied observations
            forward_sampling.sample_posterior_predictive(
                model(), trace=trace, use_auto_batching=use_auto_batching_fixture
            )
    else:
        ppc = forward_sampling.sample_posterior_predictive(
            model(), trace=trace, use_auto_batching=use_auto_batching_fixture
        ).posterior_predictive
        for k, v in ppc.items():
            assert v.shape == sample_shape_fixture + core_shapes[k]


def test_sample_posterior_predictive_on_glm(
    glm_model_fixture, use_auto_batching_fixture, sample_shape_fixture
):
    model, is_vectorized_model, core_shapes = glm_model_fixture
    trace = pm.mcmc.utils.trace_to_arviz(
        {
            # The transposition of the first two axis comes from trace_to_arviz
            # that does this to the output of `sample` to get (num_chains, num_samples, ...)
            # instead of (num_samples, num_chains, ...)
            k: tf.zeros(
                (sample_shape_fixture[1], sample_shape_fixture[0]) + sample_shape_fixture[2:] + v
            )
            for k, v in core_shapes.items()
            if k not in ["model/y"]
        }
    )
    if (
        not use_auto_batching_fixture
        and not is_vectorized_model
        and (sample_shape_fixture not in [(), (1,), (1, 1)]) > 0
    ):
        with pytest.raises(Exception):
            # This can raise many types of Exceptions.
            # For example, ValueError when tfp distributions complain about
            # the parameter shapes being imcompatible or it can raise an
            # EvaluationError because the distribution shape is not compatible
            # with the supplied observations. Also, if in a @tf.function,
            # it can raise InvalidArgumentError.
            # Furthermore, in some cases, sampling may exit without errors, but
            # the resulting shapes will be wrong
            ppc = forward_sampling.sample_posterior_predictive(
                model(), trace=trace, use_auto_batching=use_auto_batching_fixture
            )
            for k, v in ppc.items():
                assert v.shape == sample_shape_fixture + core_shapes[k]
    else:
        ppc = forward_sampling.sample_posterior_predictive(
            model(), trace=trace, use_auto_batching=use_auto_batching_fixture
        ).posterior_predictive
        for k, v in ppc.items():
            assert v.shape == sample_shape_fixture + core_shapes[k]


def test_posterior_predictive_on_root_variable(use_auto_batching_fixture):
    n_obs = 5
    n_samples = 6
    n_chains = 4

    @pm.model
    def model():
        x = yield pm.Normal(
            "x",
            np.zeros(n_obs, dtype="float32"),
            1,
            observed=np.zeros(n_obs, dtype="float32"),
            conditionally_independent=True,
            reinterpreted_batch_ndims=1,
        )
        beta = yield pm.Normal("beta", 0, 1, conditionally_independent=True)
        bias = yield pm.Normal("bias", 0, 1, conditionally_independent=True)
        mu = beta[..., None] * x + bias[..., None]
        yield pm.Normal(
            "obs", mu, 1, observed=np.ones(n_obs, dtype="float32"), reinterpreted_batch_ndims=1,
        )

    trace = pm.mcmc.utils.trace_to_arviz(
        {
            "model/beta": tf.zeros((n_samples, n_chains), dtype="float32"),
            "model/bias": tf.zeros((n_samples, n_chains), dtype="float32"),
        }
    )
    ppc = forward_sampling.sample_posterior_predictive(
        model(), trace=trace, use_auto_batching=use_auto_batching_fixture
    ).posterior_predictive
    if not use_auto_batching_fixture:
        _, state = pm.evaluate_model_posterior_predictive(
            model(), sample_shape=(n_chains, n_samples)
        )
        assert state.untransformed_values["model/x"].numpy().shape == (n_chains, n_samples, n_obs,)
    assert ppc["model/obs"].shape == (n_chains, n_samples, n_obs)
    assert ppc["model/x"].shape == (n_chains, n_samples, n_obs)
