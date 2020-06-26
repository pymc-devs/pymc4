import functools
from typing import NamedTuple
import collections

import tensorflow as tf
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

CompoundStepResults = collections.namedtuple("CompoundStepResults", ["target_log_prob",])


class _CompoundStepTF(kernel_base.TransitionKernel):
    """
        Simple support for compound step
    """

    def __init__(self, target_log_prob_fn, make_kernel_fn: NamedTuple, kernel_kwargs, name=None):
        self._target_log_prob_fn = target_log_prob_fn
        self._make_kernel_fn = make_kernel_fn
        self._kernel_kwargs = kernel_kwargs
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            make_kernel_fn=make_kernel_fn,
            kernel_kwargs=kernel_kwargs,
            name=name,
        )

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def is_calibrated(self):
        return True

    @property
    def name(self):
        return self._parameters["name"]

    def one_step(self, state, _):
        next_state = [None for _ in range(len(state))]
        for i, make_kernel_fn in enumerate(self._make_kernel_fn):

            def _target_log_prob_fn_part(state_part, idx):
                temp_value = state[idx]
                state[idx] = state_part
                log_prob = self._target_log_prob_fn(*state)
                state[idx] = temp_value
                return log_prob

            mkf = make_kernel_fn
            kernel = mkf.kernel(
                target_log_prob_fn=functools.partial(_target_log_prob_fn_part, idx=i),
                **{**self._kernel_kwargs[i], **mkf.kernel_kwargs},
            )
            if mkf.adaptive_kernel:
                kernel = mkf.adaptive_kernel(inner_kernel=kernel, **mkf.adaptive_kwargs)
            next_state[i], _ = kernel.one_step(state[i], kernel.bootstrap_results(state[i]))

        next_target_log_prob = self.target_log_prob_fn(*next_state)
        return [next_state, CompoundStepResults(target_log_prob=next_target_log_prob)]

    def bootstrap_results(self, init_state):
        with tf.name_scope(mcmc_util.make_name(self.name, "compound", "bootstrap_results")):
            if not mcmc_util.is_list_like(init_state):
                init_state = [init_state]
            init_state = [tf.convert_to_tensor(x) for x in init_state]
            init_target_log_prob = self.target_log_prob_fn(*init_state)
        return CompoundStepResults(target_log_prob=init_target_log_prob)
