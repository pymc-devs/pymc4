"""
Partially derived from a prototype written by Josh Safyan.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def sample(model,  # pylint: disable-msg=too-many-arguments
           num_results=5000,
           num_burnin_steps=3000,
           step_size=.4,
           num_leapfrog_steps=3,
           numpy=True):
    initial_state = []
    for name, (_, shape, _) in model.unobserved.items():
        initial_state.append(.5 * tf.ones(shape, name="init_{}".format(name)))

    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=model.target_log_prob_fn(),
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps))

    if numpy:
        with tf.Session() as sess:
            states, is_accepted_ = sess.run([states, kernel_results.is_accepted])
            accepted = np.sum(is_accepted_)
            print("Acceptance rate: {}".format(accepted / num_results))
    return dict(zip(model.unobserved.keys(), states))
