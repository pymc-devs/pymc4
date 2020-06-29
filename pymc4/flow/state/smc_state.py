import itertools
import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static

from .state import SamplingState


class SMCSamplingState(SamplingState):
    # TODO: make less verbose parent
    __slots__ = ()

    def _collect_log_prob_elemwise_lp_prior(self, distrs):
        """
            Collects log probabilities for prior variables in sMC
        """

        def reduce_sum_smc(value, ind_exclude):
            # `ind_exluce` argument determines the index that
            # should be kept when reducing `value` tensor.
            ndim = len(value.shape)
            return prefer_static.reduce_sum(
                value, axis=[*range(0, ind_exclude), *range(ind_exclude + 1, ndim)]
            )

        for name, dist in distrs.items():
            # TODO: for now if we are setting explicit batch dim in sMC and
            # distribution has plate parameter (batch_stack in pymc4) then the distribution
            # expects the sample of shape: [*batch_stack, draws, *event_shape] for `log_prob`
            # but the values are stored as [draws,...] because of the sequential_monte_carlo
            # op design in tfp. For now we can only check for batch_stack in distribution,
            # transpose the value and set correct axises to apply `reduce_sum`
            # This potentially can be solved with the additional argument passed to the
            # distribution initialization alongside the `batch_stack` for the BatchStacker.
            value = self.all_values[name]
            val_ndim = len(value.shape)
            int_exclude = 0
            if dist.batch_stack:
                bs = 1 if isinstance(dist.batch_stack, int) else len(dist.batch_stack)
                value = tf.transpose(value, [*range(1, bs + 1), 0, *range(bs + 1, val_ndim)])
                int_exclude = bs
            yield reduce_sum_smc(dist.log_prob(value), ind_exclude=int_exclude)

    def _collect_log_prob_elemwise_lp_lkh(self, distrs):
        for name, dist in distrs.items():
            # For sMC we need to move first axis (draws axis) to the right position
            # so that the shapes of distribution parameters and value can be broadcasted
            value_st = self.all_values[name]
            val_dims, val_ndim = value_st.shape, value_st.ndim
            dist_ndim = dist.batch_shape.rank + dist.event_shape.rank
            ndim_diff = prefer_static.abs(val_ndim - dist_ndim)

            # if...else supported by tf.function
            if val_ndim > dist_ndim:
                # If value ndim is larger thatn ndim of distribution params then
                # we are transposing the value tensor
                min_diff_ndim = prefer_static.minimum(ndim_diff + 1, val_ndim)
                axis_transpose = prefer_static.concat(
                    [
                        prefer_static.range(1, min_diff_ndim),
                        tf.constant([0]),
                        prefer_static.range(min_diff_ndim, val_ndim),
                    ],
                    axis=0,
                )
                value = tf.transpose(value_st, axis_transpose)
                reduce_axis = prefer_static.concat(
                    [
                        prefer_static.range(0, min_diff_ndim - 1),
                        prefer_static.range(min_diff_ndim, val_ndim),
                    ],
                    axis=0,
                )
            elif val_ndim < dist_ndim:
                # Otherwise we are reshaping the value tensor to make it broadcastable
                shape_ = prefer_static.concat(
                    [
                        tf.constant([val_dims[0]]),
                        prefer_static.ones(ndim_diff, dtype=tf.int32),  # indices are int32
                        tf.constant([*val_dims[1:]], dtype=tf.int32),
                    ],
                    axis=0,
                )
                value = prefer_static.reshape(value_st, shape=shape_)
                reduce_axis = prefer_static.range(1, dist_ndim)
            else:
                reduce_axis = prefer_static.range(1, val_ndim)
                value = value_st

            yield prefer_static.reduce_sum(
                dist.log_prob(value), axis=reduce_axis,
            )

    def collect_log_prob_smc(self, is_prior):
        """
            Collects log probabilities for likelihood variables in sMC.
            Since sMC requires the `draws` dimension to be kept explicitly
            while the graph is evaluated, we can't combine sMC prbability
            collection with the NUTS log probability collection.
        """
        # Because of the different logic with reduce sum
        # it is not possible to merge two log prob ops
        # so for now we are separating them with the logic
        # of executing smc log ops and sampling log prob ops
        if is_prior is True:
            distrs = self.prior_distributions
            log_prob_elemwise = self._collect_log_prob_elemwise_lp_prior(distrs)
        else:
            distrs = self.likelihood_distributions
            log_prob_elemwise = self._collect_log_prob_elemwise_lp_lkh(distrs)
            log_prob_elemwise = itertools.chain(
                log_prob_elemwise, (p.value for p in self.potentials)
            )
        log_prob = sum(log_prob_elemwise)
        return log_prob
