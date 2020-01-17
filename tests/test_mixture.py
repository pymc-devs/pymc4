import itertools
import pytest
import pymc4 as pm
import numpy as np
import tensorflow as tf


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


@pytest.fixture(scope="session", params=[0, 1])
def batch_ndim(request):
    return request.param


@pytest.fixture(scope="session", params=[2])
def mx_nc(request):
    """
        number of components for mixture distribution
    """
    return request.param


@pytest.fixture(scope="module")
def normal_mixture_params(request):
    # for now we are testing on a determenistic input
    N = 1000
    w = array([0.35, 0.4, 0.25])
    mu = array([0.0, 2.0, 5.0])
    sigma = array([0.5, 0.5, 1.0])
    return (N, w, mu, sigma)


@pytest.fixture(scope="module", params=list(itertools.product([10], [1, 2], [10], [0.1])))
def stat_params(request):
    """
        return tuple: (num_samples, num_chains, burn_in, step_size)
    """
    stat_params = request.param
    return stat_params


@pytest.fixture(scope="function", params=range(13, 20))
def mixture_distribution(request, batch_ndim, mx_nc):
    seed = request.param
    np.random.seed(seed)

    p = np.random.random((1,) * batch_ndim + (mx_nc,))
    p = p / p.sum(axis=-1)
    dists = [pm.Normal.dist(tf.zeros((1,) * batch_ndim), 1) for _ in range(mx_nc)]

    return pm.Mixture.dist(p=p, distributions=dists)


@pytest.fixture(scope="module")
def simple_mixture_model(batch_ndim, mx_nc):
    @pm.model
    def simple_mixture_model():
        p = yield pm.Dirichlet("p", concentration=tf.ones((1,) * batch_ndim + (mx_nc,)))
        dists = [pm.Normal.dist(tf.zeros((1,) * batch_ndim), 1) for _ in range(mx_nc)]
        x_obs = yield pm.Mixture("x_obs", p=p, distributions=dists)
        return x_obs

    return simple_mixture_model


@pytest.fixture(scope="module")
def mixture_model_conditioned(batch_ndim, mx_nc):
    @pm.model
    def mixture_model_conditioned():
        mean = yield pm.Normal("mean", 0, tf.ones((1,) * batch_ndim))
        p = yield pm.Dirichlet("p", concentration=tf.ones((1,) * batch_ndim + (mx_nc,)))
        dists = [pm.Normal.dist(mean, 1) for _ in range(mx_nc)]
        x_obs = yield pm.Mixture("x_obs", p=p, distributions=dists)
        return x_obs

    return mixture_model_conditioned


@pytest.fixture(scope="module")
def normal_mixture(normal_mixture_params):
    N, w, mu, sigma = normal_mixture_params
    component = np.random.choice(mu.size, size=N, p=w)
    x = np.random.normal(mu[component], sigma[component], size=N)

    @pm.model
    def normal_mixture():
        w = yield pm.Dirichlet(
            "w",
            concentration=[1, 1, 1],
            transform=pm.transforms.Invert(pm.transforms.SoftmaxCentered()),
        )
        loc = yield pm.Normal("mu", [0, 0, 0], 10.0)  # TODO: fix after `#193`
        scale = yield pm.Normal("tau", [1, 1, 1], 1.0)
        x_obs = yield pm.NormalMixture("x_obs", w=w, loc=loc, scale=scale, observed=x)
        return x_obs

    return normal_mixture, normal_mixture_params, x


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


def test_simple_mixture_model_shape(
    simple_mixture_model, xla_fixture, stat_params, batch_ndim, mx_nc
):
    model = simple_mixture_model()
    num_samples, num_chains, burn_in, step_size = stat_params
    trace, stats = pm.sample(
        model=model,
        num_samples=num_samples,
        num_chains=num_chains,
        burn_in=burn_in,
        step_size=step_size,
        xla=xla_fixture,
    )
    scope_obs = "simple_mixture_model/x_obs"
    scope_p = "simple_mixture_model/p"

    np.testing.assert_equal(
        (num_samples, num_chains) + (1,) * batch_ndim, tuple(trace[scope_obs].shape)
    )
    np.testing.assert_equal(
        (num_samples, num_chains) + (1,) * batch_ndim + (mx_nc,), tuple(trace[scope_p].shape)
    )


def test_mixture_conditioned(
    mixture_model_conditioned, xla_fixture, stat_params, batch_ndim, mx_nc
):
    model = mixture_model_conditioned()
    num_samples, num_chains, burn_in, step_size = stat_params
    trace, stats = pm.sample(
        model=model,
        num_samples=num_samples,
        num_chains=num_chains,
        burn_in=burn_in,
        step_size=step_size,
        xla=xla_fixture,
    )
    scope_obs = "mixture_model_conditioned/x_obs"
    scope_p = "mixture_model_conditioned/p"

    np.testing.assert_equal(
        (num_samples, num_chains) + (1,) * batch_ndim, tuple(trace[scope_obs].shape)
    )
    np.testing.assert_equal(
        (num_samples, num_chains) + (1,) * batch_ndim + (mx_nc,), tuple(trace[scope_p].shape)
    )


@pytest.mark.slow
@pytest.mark.xfail
def test_normal_mixture(normal_mixture):
    model, normal_mixture_params, x = normal_mixture
    model = model()
    N, w, mu, sigma = normal_mixture_params
    trace, stats = pm.sample(
        model=model, num_samples=2000, num_chains=2, burn_in=300, step_size=0.1, xla=True
    )
    np.testing.assert_allclose(
        mu,
        tf.reduce_mean(
            tf.sort(tf.reduce_mean(trace["normal_mixture/mu"], axis=0), axis=-1), axis=0
        ),
        rtol=0.1,
        atol=0.1,
    )
