import itertools
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import ChainMap
from pymc4.distributions import distribution
import tensorflow as tf

from pymc4 import utils


def _chain_map_iter(self):
    """Keep ordering of maps on Python3.6.
    See https://bugs.python.org/issue32792
    Once Python3.6 is not supported, this can be deleted.
    """
    d = {}
    for mapping in reversed(self.maps):
        d.update(mapping)  # reuses stored hash values if possible
    return iter(d)


ChainMap.__iter__ = _chain_map_iter  # type: ignore


class SamplingState:
    __slots__ = (
        "transformed_values",
        "untransformed_values",
        "transformed_values_batched",
        "untransformed_values_batched",
        "observed_values",
        "posterior_predictives",
        "all_values",
        "all_unobserved_values",
        "all_unobserved_values_batched",
        "discrete_distributions",
        "continuous_distributions",
        "prior_distributions",
        "likelihood_distributions",
        "potentials",
        "deterministics",
        "deterministics_values",
    )

    def __init__(
        self,
        transformed_values: Dict[str, Any] = None,
        untransformed_values: Dict[str, Any] = None,
        observed_values: Dict[str, Any] = None,
        discrete_distributions: Dict[str, distribution.Distribution] = None,
        continuous_distributions: Dict[str, distribution.Distribution] = None,
        potentials: List[distribution.Potential] = None,
        deterministics: Dict[str, distribution.Deterministic] = None,
        posterior_predictives: Optional[Set[str]] = None,
        transformed_values_batched: Dict[str, Any] = None,
        untransformed_values_batched: Dict[str, Any] = None,
        deterministics_values: Dict[str, Any] = None,
    ) -> None:
        # TODO: verbose __init__
        if transformed_values is None:
            transformed_values = dict()
        else:
            transformed_values = transformed_values.copy()
        if untransformed_values is None:
            untransformed_values = dict()
        else:
            untransformed_values = untransformed_values.copy()
        if transformed_values_batched is None:
            transformed_values_batched = dict()
        else:
            transformed_values_batched = transformed_values_batched.copy()
        if untransformed_values_batched is None:
            untransformed_values_batched = dict()
        else:
            untransformed_values_batched = untransformed_values_batched.copy()
        if observed_values is None:
            observed_values = dict()
        else:
            observed_values = observed_values.copy()
        if discrete_distributions is None:
            discrete_distributions = dict()
        else:
            discrete_distributions = discrete_distributions.copy()
        if continuous_distributions is None:
            continuous_distributions = dict()
        else:
            continuous_distributions = continuous_distributions.copy()
        if potentials is None:
            potentials = list()
        else:
            potentials = potentials.copy()
        if deterministics is None:
            deterministics = dict()
        else:
            deterministics = deterministics.copy()
        if posterior_predictives is None:
            posterior_predictives = set()
        else:
            posterior_predictives = posterior_predictives.copy()
        if deterministics_values is None:
            deterministics_values = dict()
        else:
            deterministics_values = deterministics_values.copy()
        self.transformed_values = transformed_values
        self.untransformed_values = untransformed_values
        self.observed_values = observed_values

        self.transformed_values_batched = transformed_values_batched
        self.untransformed_values_batched = untransformed_values_batched

        self.all_values = ChainMap(
            self.untransformed_values, self.transformed_values, self.observed_values
        )

        self.all_unobserved_values = ChainMap(self.untransformed_values, self.transformed_values)
        self.all_unobserved_values_batched = ChainMap(
            self.transformed_values_batched, self.untransformed_values_batched
        )

        self.discrete_distributions = discrete_distributions
        self.continuous_distributions = continuous_distributions
        self.prior_distributions: Dict[str, distribution.Distribution] = {}
        self.likelihood_distributions: Dict[str, distribution.Distribution] = {}

        self.potentials = potentials
        self.deterministics = deterministics
        self.posterior_predictives = posterior_predictives
        self.deterministics_values = deterministics_values

    def collect_log_prob_elemwise(self):
        return itertools.chain(
            (
                dist.log_prob(self.all_values[name])
                for name, dist in itertools.chain(
                    self.discrete_distributions.items(),
                    self.continuous_distributions.items(),
                )
            ),
            (p.value for p in self.potentials),
        )

    def collect_log_prob(self):
        return sum(map(tf.reduce_sum, self.collect_log_prob_elemwise()))

    def collect_unreduced_log_prob(self):
        return sum(self.collect_log_prob_elemwise())

    def __repr__(self):
        def get_distribution_class_names(distribution_dict):
            return ["{}:{}".format(d.__class__.__name__, k) for k, d in distribution_dict.items()]

        # display keys only
        untransformed_values = list(self.untransformed_values)
        transformed_values = list(self.transformed_values)
        observed_values = list(self.observed_values)
        deterministics = list(self.deterministics)
        posterior_predictives = list(self.posterior_predictives)
        deterministics_values = list(self.deterministics_values)
        # format like dist:name prior_distribution, likelihood_distribution
        prior_distributions = get_distribution_class_names(self.prior_distributions)
        likelihood_distributions = get_distribution_class_names(self.likelihood_distributions)
        # be less verbose here
        num_potentials = len(self.potentials)

        discrete_distributions = [
            "{}:{}".format(d.__class__.__name__, k) for k, d in self.discrete_distributions.items()
        ]
        # continuous case
        continuous_distributions = [
            "{}:{}".format(d.__class__.__name__, k)
            for k, d in self.continuous_distributions.items()
        ]

        indent = 4 * " "
        return (
            "{}(\n"
            + indent
            + "untransformed_values: {}\n"
            + indent
            + "transformed_values: {}\n"
            + indent
            + "observed_values: {}\n"
            + indent
            + "discrete_distributions: {}\n"
            + indent
            + "continuous_distributions: {}\n"
            + indent
            + "prior_distributions: {}\n"
            + indent
            + "likelihood_distributions: {}\n"
            + indent
            + "num_potentials={}\n"
            + indent
            + "deterministics: {}\n"
            + indent
            + "deterministics_values: {}\n"
            + indent
            + "posterior_predictives: {})"
        ).format(
            self.__class__.__name__,
            untransformed_values,
            transformed_values,
            observed_values,
            discrete_distributions,
            continuous_distributions,
            prior_distributions,
            likelihood_distributions,
            num_potentials,
            deterministics,
            deterministics_values,
            posterior_predictives,
        )

    @classmethod
    def from_values(
        cls, values: Dict[str, Any] = None, observed_values: Dict[str, Any] = None
    ) -> "SamplingState":
        if values is None:
            return cls(observed_values=observed_values)
        transformed_values = dict()
        untransformed_values = dict()
        # split by `nest/name` or `nest/__transform_name`
        for fullname in values:
            namespec = utils.NameParts.from_name(fullname)
            if namespec.is_transformed:
                transformed_values[fullname] = values[fullname]
            else:
                untransformed_values[fullname] = values[fullname]
        return cls(
            transformed_values,
            untransformed_values,
            observed_values,
            transformed_values_batched=dict(),
            untransformed_values_batched=dict(),
        )

    def clone(self) -> "SamplingState":
        return self.__class__(
            transformed_values=self.transformed_values,
            untransformed_values=self.untransformed_values,
            observed_values=self.observed_values,
            discrete_distributions=self.discrete_distributions,
            continuous_distributions=self.continuous_distributions,
            potentials=self.potentials,
            deterministics=self.deterministics,
            posterior_predictives=self.posterior_predictives,
            deterministics_values=self.deterministics_values,
        )

    def as_sampling_state(self) -> Tuple["SamplingState", List[str]]:
        """Create a sampling state that should be used within MCMC sampling.
        There are some principles that hold for the state.
            1. Check there is at least one distribution
            2. Check all transformed distributions are autotransformed
            3. Remove untransformed values if transformed are present
            4. Remove all other irrelevant values
        """
        if not self.discrete_distributions and not self.continuous_distributions:
            raise TypeError(
                "No distributions found in the state. "
                "the model you evaluated is empty and does not yield any PyMC4 distribution"
            )
        untransformed_values = dict()
        transformed_values = dict()
        need_to_transform_after = list()
        observed_values = dict()

        for name, dist in itertools.chain(
            self.discrete_distributions.items(), self.continuous_distributions.items()
        ):
            namespec = utils.NameParts.from_name(name)
            if dist.transform is not None and name not in self.observed_values:
                transformed_namespec = namespec.replace_transform(dist.transform.name)
                if transformed_namespec.full_original_name not in self.transformed_values:
                    raise TypeError(
                        "Transformed value {!r} is not found for {} distribution with name {!r}. "
                        "You should evaluate the model using the transformed executor to get "
                        "the correct sampling state.".format(
                            transformed_namespec.full_original_name, dist, name
                        )
                    )
                else:
                    transformed_values[
                        transformed_namespec.full_original_name
                    ] = self.transformed_values[transformed_namespec.full_original_name]
                    need_to_transform_after.append(transformed_namespec.full_untransformed_name)
            else:
                if name in self.observed_values:
                    observed_values[name] = self.observed_values[name]
                elif name in self.untransformed_values:
                    untransformed_values[name] = self.untransformed_values[name]
                else:
                    raise TypeError(
                        "{} distribution with name {!r} does not have the corresponding value "
                        "in the state. This may happen if the current "
                        "state was modified in the wrong way."
                    )
        return (
            self.__class__(
                transformed_values=transformed_values,
                untransformed_values=untransformed_values,
                transformed_values_batched=self.transformed_values_batched.copy(),
                untransformed_values_batched=self.untransformed_values_batched.copy(),
                observed_values=observed_values,
            ),
            need_to_transform_after,
        )
