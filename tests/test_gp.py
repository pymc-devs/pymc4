"""Test suite for GP Module"""

import tensorflow as tf

import pymc4 as pm

import pytest

# Test all the GP models only using a particular
# mean and covariance functions but varying tensor shapes
# NOTE: the mean and covariance functions used here
# must be present in `MEAN_FUNCS` and `COV_FUNCS` resp.
GP_MODELS = [
    (
        "LatentGP",
        {"mean_fn": ("Zero", {}), "cov_fn": ("ExpQuad", {"amplitude": 1.0, "length_scale": 1.0}),},
    ),
]


@pytest.fixture(scope="module", params=GP_MODELS, ids=str)
def get_gp_model(request):
    return request.param


def build_model(model_name, model_kwargs, feature_ndims):
    """Create a gp model from an element in the `GP_MODELS` list"""
    # First, create a mean function
    name = model_kwargs["mean_fn"][0]
    kwargs = model_kwargs["mean_fn"][1]
    MeanClass = getattr(pm.gp.mean, name)
    mean_fn = MeanClass(**kwargs, feature_ndims=feature_ndims)
    # Then, create the kernel function
    name = model_kwargs["cov_fn"][0]
    kwargs = model_kwargs["cov_fn"][1]
    KernelClass = getattr(pm.gp.cov, name)
    cov_fn = KernelClass(**kwargs, feature_ndims=feature_ndims)
    # Now, create the model and return
    GPModel = getattr(pm.gp, model_name)
    model = GPModel(mean_fn=mean_fn, cov_fn=cov_fn)
    return model


def test_gp_models_prior(tf_seed, get_data, get_gp_model):
    """Test the prior method of a GP mode, if present"""
    batch_shape, sample_shape, feature_shape, X = get_data
    gp_model = build_model(get_gp_model[0], get_gp_model[1], len(feature_shape))
    # @pm.model
    # def model(gp, X):
    #     yield gp.prior('f', X)
    try:
        # sampling_model = model(gp_model, X)
        # trace = pm.sample(sampling_model, num_samples=3, num_chains=1, burn_in=10)
        # trace = np.asarray(trace.posterior["model/f"])
        prior_dist = gp_model.prior("prior", X)
    except NotImplementedError:
        pytest.skip("Skipping: prior not implemented")
    # if sample_shape == (1,):
    #     assert trace.shape == (1, 3, ) + batch_shape
    # else:
    #     assert trace.shape == (1, 3, ) + batch_shape + sample_shape
    if sample_shape == (1,):
        assert prior_dist.sample(1).shape == (1,) + batch_shape
    else:
        assert prior_dist.sample(1).shape == (1,) + batch_shape + sample_shape


def test_gp_models_conditional(tf_seed, get_data, get_gp_model):
    """Test the conditional method of a GP mode, if present"""
    batch_shape, sample_shape, feature_shape, X = get_data
    gp_model = build_model(get_gp_model[0], get_gp_model[1], len(feature_shape))
    Xnew = tf.random.normal(batch_shape + sample_shape + feature_shape)

    @pm.model
    def model(gp, X, Xnew):
        f = yield gp.prior("f", X)
        yield gp.conditional("fcond", Xnew, given={"X": X, "f": f})

    try:
        # sampling_model = model(gp_model, X, Xnew)
        # trace = pm.sample(sampling_model, num_samples=3, num_chains=1, burn_in=10)
        # trace = np.asarray(trace.posterior["model/fcond"])
        f = gp_model.prior("f", X).sample(1)[0]
        cond_dist = gp_model.conditional("fcond", Xnew, given={"X": X, "f": f})
        cond_samples = cond_dist.sample(3)
    except NotImplementedError:
        pytest.skip("Skipping: conditional not implemented")
    # if sample_shape == (1,):
    #     assert trace.shape == (1, 3,) + batch_shape
    # else:
    #     assert trace.shape == (1, 3,) + batch_shape + sample_shape
    if sample_shape == (1,):
        assert cond_samples.shape == (3,) + batch_shape
    else:
        assert cond_samples.shape == (3,) + batch_shape + sample_shape
