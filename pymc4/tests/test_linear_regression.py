import pymc4 as pm
import tensorflow as tf
import numpy as np
import pytest
from numpy import array, float32


@pytest.fixture(scope="function")
def linear_regression(tf_session):
    """"""
    # Logp calculation for linear regression
    @pm.model
    def linreg(n_points=100):
        # Define priors
        sigma = pm.HalfNormal("sigma", sigma=10)
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        x_coeff = pm.Normal("weight", mu=0, sigma=5)
        x = np.linspace(-5, 5, n_points)

        # Define likelihood
        y = pm.Normal("y", mu=intercept + x_coeff * x, sigma=sigma)

    model = linreg.configure()
    forward_sample = tf_session.run(model.forward_sample())

    return model, forward_sample, tf_session


@pytest.mark.xfail(reason="Sampling is now done in transformed space?")
def test_linear_regression(linear_regression):
    model, forward_sample, tf_session = linear_regression
    sigma = tf.placeholder(tf.float32)
    intercept = tf.placeholder(tf.float32)
    x_coeff = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, shape=(100,))
    func = model.make_log_prob_function()

    logp = func(sigma, intercept, x_coeff, y)
    feed_dict = {
        sigma: forward_sample["sigma"],
        intercept: forward_sample["intercept"],
        x_coeff: forward_sample["weight"],
        y: forward_sample["y"],
    }
    np.testing.assert_allclose(tf_session.run(logp, feed_dict=feed_dict), -437.2295)


@pytest.fixture
def fixed_forward_samples():
    fixed_forward_samples = {
        "intercept": 10.935_236,
        "sigma": 16.7994,
        "weight": 8.037_117,
        "y": array(
            [
                -24.576_231,
                -22.682_533,
                -36.978_157,
                -13.811_799,
                -44.940_895,
                -37.86523,
                -29.78477,
                -20.712_181,
                -18.465_843,
                -22.156_605,
                -10.810_569,
                -26.692_762,
                -0.245_002_75,
                -43.92573,
                -32.912_354,
                2.942_428_6,
                -6.474_824,
                9.213_276,
                -6.553_390_5,
                -40.98478,
                8.841_461,
                -2.425_013_5,
                -7.318_285,
                1.626_349_4,
                7.605_583,
                -10.515_693,
                13.96204,
                -16.085_886,
                2.013_514_5,
                0.689_804_1,
                -7.677_414,
                -6.887_258_5,
                -24.356_915,
                5.995_925,
                -39.95355,
                -23.149_841,
                -19.996_231,
                -40.76818,
                -16.252_708,
                18.481_174,
                7.057_470_3,
                9.800_463,
                28.474_716,
                4.31737,
                18.575_928,
                -3.294_704_4,
                14.653_564,
                24.271_791,
                24.279_163,
                18.800_606,
                29.928_093,
                12.044_709,
                35.58055,
                28.914_322,
                9.351_996,
                25.00039,
                22.58787,
                12.904_894,
                -11.597_815,
                12.835_809,
                -18.172_718,
                9.920_753,
                28.220_247,
                12.237_292,
                -23.608_627,
                38.045_254,
                22.484_333,
                -4.954_643_2,
                1.350_353_2,
                23.522_764,
                34.169_632,
                14.337_034,
                32.802_563,
                14.787_531,
                14.242_445,
                34.308_456,
                11.320_705,
                70.35381,
                22.60202,
                39.99462,
                31.371_067,
                17.169_744,
                17.658_762,
                55.54461,
                64.28799,
                20.262_682,
                18.32425,
                38.043_655,
                27.912_197,
                59.457_718,
                43.137_627,
                29.909_275,
                71.459_656,
                31.067_802,
                31.44457,
                74.863_266,
                38.846_256,
                70.65802,
                55.643_463,
                71.11842,
            ],
            dtype=float32,
        ),
    }
    return fixed_forward_samples


@pytest.mark.xfail(reason="Sampling is now donei n transformed space?")
def test_linear_regression_forward_sample(linear_regression, fixed_forward_samples):
    model, forward_sample, tf_session = linear_regression
    np.testing.assert_almost_equal(forward_sample["y"], fixed_forward_samples["y"], decimal=2)
    np.testing.assert_almost_equal(
        forward_sample["sigma"], fixed_forward_samples["sigma"], decimal=2
    )
    np.testing.assert_almost_equal(
        forward_sample["intercept"], fixed_forward_samples["intercept"], decimal=1
    )
    np.testing.assert_almost_equal(
        forward_sample["weight"], fixed_forward_samples["weight"], decimal=2
    )
