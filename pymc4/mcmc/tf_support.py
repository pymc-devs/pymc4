import functools
from typing import NamedTuple
import collections
import numpy as np

import tensorflow as tf
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

CompoundStepResults = collections.namedtuple("CompoundStepResults", ["compound_results"])


def _target_log_prob_fn_part(*state_part, idx, len_, state, target_log_prob_fn):
    sl = slice(idx, idx + len_)
    temp_value = state[sl]
    state[sl] = state_part
    log_prob = target_log_prob_fn(*state)
    state[sl] = temp_value
    return log_prob


def kernel_create_object():
    # TODO: we can replace kernel creation logic
    ...


class _CompoundStepTF(kernel_base.TransitionKernel):
    """
        Simple support for compound step
        TODO:
    """

    def __init__(self, target_log_prob_fn, make_kernel_fn: NamedTuple, kernel_kwargs, li, name=None):
        self._target_log_prob_fn = target_log_prob_fn
        self._make_kernel_fn = make_kernel_fn
        self._kernel_kwargs = kernel_kwargs
        # TODO: order could be wrong, sort in samplers.py
        self.li = li
        self._name = name
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            make_kernel_fn=make_kernel_fn,
            kernel_kwargs=kernel_kwargs,
            name=name,
        )

    @property
    def target_log_prob_fn(self):
        return self._target_log_prob_fn

    @property
    def parameters(self):
        return self._parameters

    @property
    def name(self):
        return self._name

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results):
        """Takes one step of the TransitionKernel
        TODO: More specific fore compound step
        """
        with tf.name_scope(mcmc_util.make_name(self.name, "compound", "one_step")):
            unwrap_state_list = not tf.nest.is_nested(current_state)
            if unwrap_state_list:
                current_state = [current_state]
            next_state = []
            next_results = []
            previous_kernel_results = previous_kernel_results.compound_results
            for i, (make_kernel_fni, resulti, kwargsi, li) in enumerate(
                zip(self._make_kernel_fn, previous_kernel_results, self._kernel_kwargs, self.li)
            ):
                mkf = make_kernel_fni
                kernel = mkf.kernel(
                    target_log_prob_fn=functools.partial(
                        _target_log_prob_fn_part,
                        idx=i,
                        len_=li,
                        state=current_state,
                        target_log_prob_fn=self._target_log_prob_fn,
                    ),
                    **{**kwargsi, **mkf.kernel_kwargs},
                )
                if mkf.adaptive_kernel:
                    kernel = mkf.adaptive_kernel(inner_kernel=kernel, **mkf.adaptive_kwargs)
                next_state_, next_result_ = kernel.one_step(
                    current_state[slice(i, i + li)], resulti
                )
                # concat state results for flattened list
                next_state += next_state_
                # save current results
                next_results.append(next_result_)
        return [next_state, CompoundStepResults(compound_results=next_results)]

    def bootstrap_results(self, init_state):
        """Returns an object with the same type as returned by `one_step(...)[1]`
        Compound bootrstrap step
        """
        with tf.name_scope(mcmc_util.make_name(self.name, "compound", "bootstrap_results")):
            if not mcmc_util.is_list_like(init_state):
                init_state = [init_state]
            init_state = [tf.convert_to_tensor(x) for x in init_state]

            init_results = []
            for i, (make_kernel_fn, kwargsi, li) in enumerate(
                zip(self._make_kernel_fn, self._kernel_kwargs, self.li)
            ):
                mkf = make_kernel_fn
                kernel = mkf.kernel(
                    target_log_prob_fn=functools.partial(
                        _target_log_prob_fn_part,
                        idx=i,
                        len_=li,
                        state=init_state,
                        target_log_prob_fn=self._target_log_prob_fn,
                    ),
                    **{**kwargsi, **mkf.kernel_kwargs},
                )
                if mkf.adaptive_kernel:
                    kernel = mkf.adaptive_kernel(inner_kernel=kernel, **mkf.adaptive_kwargs)
                # bootstrap results in list
                init_results.append(kernel.bootstrap_results(init_state[slice(i, i + li)]))
        return CompoundStepResults(compound_results=init_results)
