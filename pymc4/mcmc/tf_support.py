import functools
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


def kernel_create_object(sampleri, curr_indx, setli, current_state, target_log_prob_fn):
    mkf = sampleri[0]
    kernel = mkf.kernel(
        target_log_prob_fn=functools.partial(
            _target_log_prob_fn_part,
            idx=curr_indx,
            len_=setli,
            state=current_state,
            target_log_prob_fn=target_log_prob_fn,
        ),
        **{**sampleri[1], **mkf.kernel_kwargs},
    )
    if mkf.adaptive_kernel:
        kernel = mkf.adaptive_kernel(inner_kernel=kernel, **mkf.adaptive_kwargs)
    return kernel


class _CompoundStepTF(kernel_base.TransitionKernel):
    def __init__(
        self, target_log_prob_fn, compound_samplers, compound_set_lengths, name=None,
    ):
        """
           Initializes the compound step transition kernel

           Args:
             target_log_prob_fn: Python callable which takes an argument like
               `current_state` (or `*current_state` if it's a list) and returns its
               (possibly unnormalized) log-density under the target distribution.
             compound_samplers: List of the pymc4.mcmc.samplers._BaseSampler sub-classes
               that are used for each subset of the free variables
             compound_set_lengths: List of the sizes of each subset of variables
     >>      seed: Python integer to seed the random number generator. Deprecated, pass
               seed to `tfp.mcmc.sample_chain`.
             name: Python `str` name prefixed to Ops created by this function.
        """

        with tf.name_scope(name or "CompoundSampler") as name:
            self._target_log_prob_fn = target_log_prob_fn
            self._compound_samplers = [
                (sampler[0]._default_kernel_maker(), sampler[1]) for sampler in compound_samplers
            ]
            self._compound_set_lengths = compound_set_lengths
            self._cumulative_lengths = np.cumsum(compound_set_lengths) - compound_set_lengths
            self._name = name
            self._parameters = dict(target_log_prob_fn=target_log_prob_fn, name=name)

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

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Takes one step of the TransitionKernel
        TODO: More specific fore compound step
        """
        with tf.name_scope(mcmc_util.make_name(self.name, "compound", "one_step")):
            unwrap_state_list = not tf.nest.is_nested(current_state)
            if unwrap_state_list:
                current_state = [current_state]
            # TODO: can't use any efficient type of structure here
            # like tf.TensorArray..., `next_state` can have multiple tf.dtype and
            # `next_results` stores namedtuple with the set of tf.Tensor's with
            # multiple tf.dtype too.
            next_state = []
            next_results = []
            previous_kernel_results = previous_kernel_results.compound_results

            for sampleri, setli, resulti, curri in zip(
                self._compound_samplers,
                self._compound_set_lengths,
                previous_kernel_results,
                self._cumulative_lengths,
            ):
                kernel = kernel_create_object(
                    sampleri, curri, setli, current_state, self._target_log_prob_fn
                )
                next_state_, next_result_ = kernel.one_step(
                    current_state[slice(curri, curri + setli)], resulti, seed=seed
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
            for sampleri, setli, curri in zip(
                self._compound_samplers, self._compound_set_lengths, self._cumulative_lengths
            ):
                kernel = kernel_create_object(
                    sampleri, curri, setli, init_state, self._target_log_prob_fn
                )
                # bootstrap results in listj
                init_results.append(
                    kernel.bootstrap_results(init_state[slice(curri, curri + setli)])
                )

        return CompoundStepResults(compound_results=init_results)
