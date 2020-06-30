import itertools
from tensorflow_probability.python.internal import prefer_static
from .state import SamplingState


class SMCSamplingState(SamplingState):
    """
        TODO: #229
    """

    __slots__ = ()

    def _collect_log_prob_elemwise(self, distrs):
        """
            Collects log probabilities for prior/likelihood variables in sMC
        """
        return [
            prefer_static.reduce_sum(dist.log_prob(self.all_values[name]))
            for name, dist in distrs.items()
        ]

    def collect_log_prob_smc(self, is_prior):
        """
            Collects log probabilities for likelihood variables in sMC.
            Since sMC requires the `draws` dimension to be kept explicitly
            while the graph is evaluated, we can't combine sMC prbability
            collection with the NUTS log probability collection.
        """
        if is_prior is True:
            distrs = self.prior_distributions
            log_prob_elemwise = self._collect_log_prob_elemwise(distrs)
        else:
            distrs = self.likelihood_distributions
            log_prob_elemwise = self._collect_log_prob_elemwise(distrs)
            log_prob_elemwise = itertools.chain(
                log_prob_elemwise, (p.value for p in self.potentials)
            )
        log_prob = sum(log_prob_elemwise)
        return log_prob
