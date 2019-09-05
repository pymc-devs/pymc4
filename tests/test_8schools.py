import pymc4 as pm4
import numpy as np

J = 8
y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)


@pm4.model
def schools_pm4():
    eta = yield pm4.Normal("eta", 0, 1, plate=J)
    mu = yield pm4.Normal("mu", 1, 1e6)
    tau = yield pm4.HalfNormal("tau", 1 * 2.0)

    theta = mu + tau * eta

    obs = yield pm4.Normal("obs", theta, sigma=sigma, observed=y)
    return obs


def test_sample_no_xla():
    # TODO: better test, compare to etalon chain from pymc3,
    #   for now it is only to veryfy it is runnable
    tf_trace = pm4.inference.sampling.sample(
        schools_pm4(), step_size=0.28, num_chains=4, num_samples=100, burn_in=50, xla=False
    )
    print(tf_trace)
