import pymc4 as pm
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
tf.random.set_random_seed(37208)  # random.org
config.intra_op_parallelism_threads = 1
sess = tf.Session(config=config)

# Logp calculation for linear regression
@pm.model
def linreg(n_points=100):
    # Define priors
    sigma = pm.HalfNormal("sigma", scale=10)
    intercept = pm.Normal("intercept", 0, scale=10)
    x_coeff = pm.Normal("weight", 0, scale=5)
    x = np.linspace(-5, 5, n_points)

    # Define likelihood
    y = pm.Normal("y", loc=intercept + x_coeff * x, scale=sigma)


model = linreg.configure()

forward_sample = sess.run(model.forward_sample())

forward_sample


func = model.make_log_prob_function()
sigma = tf.placeholder(tf.float32)
intercept = tf.placeholder(tf.float32)
x_coeff = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, shape=(100,))
logp = func(sigma, intercept, x_coeff, y)


def test_linear_regression():
    feed_dict = {
        sigma: forward_sample["sigma"],
        intercept: forward_sample["intercept"],
        x_coeff: forward_sample["weight"],
        y: forward_sample["y"],
    }
    np.testing.assert_allclose(sess.run(logp, feed_dict=feed_dict), -437.2295)
