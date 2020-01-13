import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="function", params=range(13, 15))
def mixture_distribution(request):
    if request.param == 13:
        p = np.random.random((2))
        p = p / p.sum(axis=-1)
        dists = [pm.Normal.dist(0, 1), pm.Normal.dist(0, 1)]
    else:
        p = np.random.random((2, 2))
        p = p / p.sum(axis=-1)
        dists = [pm.Normal.dist(0, [0, 1]), pm.Normal.dist(0, [0, 2])]
    return pm.Mixture.dist(p=p, distributions=dists)


@pytest.fixture(scope="function")
def simple_mixture_model():
    @pm.model
    def simple_mixture_model():
        p = yield pm.Dirichlet("p", concentration=tf.constant([1.0, 1.0]))
        dists = [pm.Normal.dist(0.0, 1.0), pm.Normal.dist(0.0, 1.0)]
        x_obs = yield pm.Mixture("x_obs", p=p, distributions=dists)
        return x_obs

    return simple_mixture_model


@pytest.fixture(scope="function")
def mixture_model_conditioned():
    @pm.model
    def mixture_model_conditioned():
        mean = yield pm.Normal("mean", 0, tf.constant([0.0, 1.0]))
        p = yield pm.Dirichlet("p", concentration=tf.constant([[1.0, 1.0], [1.0, 1.0]]))
        dists = [pm.Normal.dist(mean, 1.0), pm.Normal.dist(mean, 1.0)]
        x_obs = yield pm.Mixture("x_obs", p=p, distributions=dists)
        return x_obs

    return mixture_model_conditioned


@pytest.fixture(scope="function")
def normal_mixture():
    @pm.model
    def normal_mixture():
        w = yield pm.Dirichlet("w", concentration=[0, 1])
        loc = yield pm.Normal("mu", 0.0, [10.0, 5.0])
        scale = yield pm.Gamma("tau", 1.0, [1.0, 2.0])
        x_obs = yield pm.NormalMixture("x_obs", w=w, loc=loc, scale=scale)
        return x_obs

    return normal_mixture


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            "XLA",
            marks=pytest.mark.xfail(reason="XLA compilation in sample is not fully supported yet"),
        ),
        "noXLA",
    ],
    ids=str,
)
def xla_fixture(request):
    return request.param == "XLA"


def test_mixture_distribution_logp(mixture_distribution, xla_fixture):
    distr = mixture_distribution
    sample = distr.sample()
    log_prob = distr.log_prob(value=sample)

    # calculate manually
    distr_tf = distr._distribution
    distribution_log_probs = tf.stack([d.log_prob(sample) for d in distr_tf.components], axis=-1)
    cat_log_probs = tf.math.log_softmax(distr_tf.cat.logits_parameter())
    sum_probs = distribution_log_probs + cat_log_probs

    np.testing.assert_allclose(tf.math.reduce_logsumexp(sum_probs, axis=-1), log_prob)


def test_simple_mixture_model(simple_mixture_model, xla_fixture):
    model = simple_mixture_model()
    trace, stats = pm.sample(
        model=model, num_samples=10, num_chains=4, burn_in=10, step_size=0.1, xla=xla_fixture
    )
    scope_obs = "simple_mixture_model/x_obs"
    scope_p = "simple_mixture_model/p"
    np.testing.assert_equal((10, 4), tuple(trace[scope_obs].shape))
    np.testing.assert_equal((10, 4, 2), tuple(trace[scope_p].shape))


def test_mixture_conditioned(mixture_model_conditioned, xla_fixture):
    model = mixture_model_conditioned()
    trace, stats = pm.sample(
        model=model, num_samples=10, num_chains=4, burn_in=10, step_size=0.1, xla=xla_fixture
    )
    scope_obs = "mixture_model_conditioned/x_obs"
    scope_p = "mixture_model_conditioned/p"
    np.testing.assert_equal((10, 4, 2), tuple(trace[scope_obs].shape))
    np.testing.assert_equal((10, 4, 2, 2), tuple(trace[scope_p].shape))


def test_normal_mixture(normal_mixture, xla_fixture):
    model = normal_mixture()
    trace, stats = pm.sample(
        model=model, num_samples=10, num_chains=2, burn_in=2, step_size=0.1, xla=xla_fixture
    )
    scope_obs = "normal_mixture/x_obs"
    np.testing.assert_equal((10, 2), tuple(trace[scope_obs].shape))
