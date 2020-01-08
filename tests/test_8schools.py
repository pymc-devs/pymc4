import pymc4 as pm4
import numpy as np
import tensorflow as tf

from numpy.testing import assert_almost_equal

J = 8
y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)


@pm4.model
def schools_pm4():
    eta = yield pm4.Normal("eta", 0, 1, plate=J)
    mu = yield pm4.Normal("mu", 1, 10)
    tau = yield pm4.HalfNormal("tau", 1 * 2.0)

    theta = mu + tau * eta

    obs = yield pm4.Normal("obs", theta, sigma, observed=y)
    return obs


def test_model_logp():
    """Make sure log probability matches standard.

    Recreate this with

    ```python
    import pymc3 as pm
    import numpy as np

    J = 8
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])


    with pm.Model() as eight_schools:
        eta = pm.Normal("eta", 0, 1, shape=J)
        mu = pm.Normal("mu", 1, 10)
        tau = pm.HalfNormal("tau", 2.0)

        theta = mu + tau * eta

        pm.Normal("obs", theta, sigma, observed=y)

    print(eight_schools.logp({'eta': np.zeros(8), 'mu': 0, 'tau_log__': 1}).astype(np.float32))
    ```
    """
    logp, *_ = pm4.inference.sampling.build_logp_and_deterministic_functions(
        schools_pm4(), observed={"obs": y}, state=None
    )
    init_value = logp(
        **{
            "schools_pm4/eta": tf.zeros(8),
            "schools_pm4/mu": tf.zeros(()),
            "schools_pm4/__log_tau": tf.ones(()),
        }
    ).numpy()

    assert_almost_equal(init_value, -42.876_114)


def test_sample_no_xla():
    # TODO: better test, compare to a golden standard chain from pymc3,
    #   for now it is only to verify it is runnable
    chains, samples = 4, 100
    trace = pm4.inference.sampling.sample(
        schools_pm4(), step_size=0.28, num_chains=chains, num_samples=samples, burn_in=50, xla=False
    ).posterior
    for var_name in ("eta", "mu", "tau", "__log_tau"):
        assert f"schools_pm4/{var_name}" in trace.keys()
