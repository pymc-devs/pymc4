import abc
import itertools
import inspect
from typing import Optional, List, Union, Any, Dict
import tensorflow as tf
from tensorflow_probability import mcmc
from pymc4.mcmc.utils import (
    initialize_sampling_state,
    trace_to_arviz,
    initialize_state,
    scope_remove_transformed_part_if_required,
    KERNEL_KWARGS_SET,
)
from functools import partial

from pymc4.coroutine_model import Model
from pymc4.utils import NameParts
from pymc4 import flow
from pymc4.mcmc.tf_support import _CompoundStepTF
import logging

MYPY = False

__all__ = ["HMC", "NUTS", "RandomWalkM", "CompoundStep", "NUTSSimple", "HMCSimple"]
reg_samplers = {}

# TODO: better design for logging
console = logging.StreamHandler()
_log = logging.getLogger("pymc4.sampling")
if not MYPY:
    logging._warn_preinit_stderr = 0
    _log.root.handlers = []  # remove tf absl logging handlers
_log.setLevel(logging.INFO)
_log.addHandler(console)


def register_sampler(cls):
    reg_samplers[cls._name] = cls
    return cls


class _BaseSampler(metaclass=abc.ABCMeta):
    _grad = False

    def __init__(
        self,
        model: Model,
        **kwargs,
    ):
        if not isinstance(model, Model):
            raise TypeError(
                "`sample` function only supports `pymc4.Model` objects, but \
                    you've passed `{}`".format(
                    type(model)
                )
            )

        _, _, disc_names, cont_names, _, _ = initialize_state(model)
        # if sampler has the gradient calculation during `one_step`
        # and the model contains discrete distributions then we throw the
        # error.
        if self._grad is True and disc_names:
            raise ValueError(
                "Discrete distributions can't be used with \
                    gradient-based sampler"
            )

        self.model = model
        self.stat_names: List[str] = []
        self.parent_inds: List[int] = []
        # assign arguments from **kwargs to distinct kwargs for
        # `kernel`, `adaptation_kernel`, `chain_sampler`
        self._assign_arguments(kwargs)
        # check arguments for correctness
        self._check_arguments()
        self._bound_kwargs()

    def _sample(
        self,
        *,
        num_samples: int = 1000,
        num_chains: int = 10,
        burn_in: int = 100,
        observed: Optional[dict] = None,
        state: Optional[flow.SamplingState] = None,
        use_auto_batching: bool = True,
        xla: bool = False,
        seed: Optional[int] = None,
        is_compound: bool = False,
        trace_discrete: Optional[List[str]] = None,
        include_log_likelihood=False,
    ):
        if state is not None and observed is not None:
            raise ValueError("Can't use both `state` and `observed` arguments")
        (
            logpfn,
            init,
            _deterministics_callback,
            deterministic_names,
            state_,
        ) = build_logp_and_deterministic_functions(
            self.model,
            num_chains=num_chains,
            state=state,
            observed=observed,
            collect_reduced_log_prob=use_auto_batching,
            parent_inds=self.parent_inds if is_compound else None,
        )

        init_state = list(init.values())
        init_keys = list(init.keys())

        if is_compound:
            init_state = [init_state[i] for i in self.parent_inds]
            init_keys = [init_keys[i] for i in self.parent_inds]

        if use_auto_batching:
            self.parallel_logpfn = vectorize_logp_function(logpfn)
            self.deterministics_callback = vectorize_logp_function(_deterministics_callback)
            init_state = tile_init(init_state, num_chains)
        else:
            self.parallel_logpfn = logpfn
            self.deterministics_callback = _deterministics_callback
            init_state = tile_init(init_state, num_chains)

        # TODO: problem with tf.function when passing as argument to self._run_chains
        self._num_samples = num_samples
        self.seed = seed

        if xla:
            results, sample_stats = tf.xla.experimental.compile(
                self._run_chains,
                inputs=[init_state, burn_in],
            )
        else:
            results, sample_stats = self._run_chains(init_state, burn_in)

        posterior = dict(zip(init_keys, results))

        if trace_discrete:
            # TODO: maybe better logic can be written here
            # The workaround to cast variables post-sample.
            # `trace_discrete` is the list of vairables that need to be casted
            # to tf.int32 after the sampling is completed.
            init_keys_ = [
                scope_remove_transformed_part_if_required(_, state_.transformed_values)[1]
                for _ in init_keys
            ]
            discrete_indices = [init_keys_.index(_) for _ in trace_discrete]
            keys_to_cast = [init_keys[_] for _ in discrete_indices]
            for key in keys_to_cast:
                posterior[key] = tf.cast(posterior[key], dtype=tf.int32)

        # Keep in sync with pymc3 naming convention
        if len(sample_stats) > len(self.stat_names):
            deterministic_values = sample_stats[len(self.stat_names) :]
            sample_stats = sample_stats[: len(self.stat_names)]
        sampler_stats = dict(zip(self.stat_names, sample_stats))
        if len(deterministic_names) > 0:
            posterior.update(dict(zip(deterministic_names, deterministic_values)))

        log_likelihood_dict = dict()
        if include_log_likelihood:
            log_likelihood_dict = calculate_log_likelihood(self.model, posterior, state_)
        return trace_to_arviz(
            posterior,
            sampler_stats if is_compound is False else None,
            observed_data=state_.observed_values,
            log_likelihood=log_likelihood_dict if include_log_likelihood else None,
        )

    @tf.function(autograph=False)
    def _run_chains(self, init, burn_in):
        kernel = self._kernel(target_log_prob_fn=self.parallel_logpfn, **self.kernel_kwargs)
        if self._adaptation:
            adapt_kernel = self._adaptation(inner_kernel=kernel, **self.adaptation_kwargs)
        else:
            adapt_kernel = kernel

        results, sample_stats = mcmc.sample_chain(
            self._num_samples,
            current_state=init,
            kernel=adapt_kernel,
            num_burnin_steps=burn_in,
            trace_fn=self.trace_fn,
            seed=self.seed,
            **self.chain_kwargs,
        )
        return results, sample_stats

    def _assign_arguments(self, kwargs):
        kwargs_keys = set(kwargs.keys())
        # fetch adaptation kernel, kernel, and `sample_chain` kwargs keys
        adaptation_keys = (
            set(list(inspect.signature(self._adaptation.__init__).parameters.keys())[1:])
            if self._adaptation
            else set()
        )
        kernel_keys = set(list(inspect.signature(self._kernel.__init__).parameters.keys())[1:])
        chain_keys = set(list(inspect.signature(mcmc.sample_chain).parameters.keys()))

        # intersection of key sets of each object from
        # (`self._adaptation`, `self._kernel`, `sample_chain`)
        # is the kwargs we are trying to find
        self.adaptation_kwargs = {k: kwargs[k] for k in (adaptation_keys & kwargs_keys)}
        self.kernel_kwargs = {k: kwargs[k] for k in (kernel_keys & kwargs_keys)}
        self.chain_kwargs = {k: kwargs[k] for k in (chain_keys & kwargs_keys)}

    def _check_arguments(self):
        # check if there is an ambiguity of the kwargs keys for
        # kernel, adaptation_kernel adn sample_chain method
        if (
            (self.adaptation_kwargs.keys() & self.kernel_kwargs.keys())
            or (self.adaptation_kwargs.keys() & self.chain_kwargs.keys())
            or (self.kernel_kwargs.keys() & self.chain_kwargs.keys())
        ):
            raise ValueError(
                "Ambiguity in setting kwargs for `kernel`, \
                        `adaptation_kernel`, `chain_sampler`"
            )

    def _bound_kwargs(self, *args):
        # set all the default kwargs which are distinct
        # for each type of sampler. If a use has passed
        # the key argument then we don't change the kwargs set
        for k, v in self.default_kernel_kwargs.items():
            self.kernel_kwargs.setdefault(k, v)
        for k, v in self.default_adapter_kwargs.items():
            self.adaptation_kwargs.setdefault(k, v)

    def __call__(self, *args, **kwargs):
        # pm.sample() entrance
        return self._sample(*args, **kwargs)

    @classmethod
    def _default_kernel_maker(cls):
        # The function is used for compound step support.
        # by supporting collection we could easily instantiate
        # kernel inside the `one_step`
        # TODO: maybe can be done with partial, but not
        # sure how to do it recursively
        kernel_collection = KERNEL_KWARGS_SET(
            kernel=cls._kernel,
            adaptive_kernel=cls._adaptation,
            kernel_kwargs=cls.default_kernel_kwargs,
            adaptive_kwargs=cls.default_adapter_kwargs,
        )
        return kernel_collection

    @abc.abstractmethod
    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        """
        Support a tracing for each sampler

        Parameters
        ----------
        current_state : flow.SamplingState
            state for tracing
        pkr : Union[tf.Tensor, Any]
            A `Tensor` or nested collection of `Tensor`s
        """
        pass


@register_sampler
class HMC(_BaseSampler):
    """
    Sampler to run one_step of Hamiltonian Monte Carlo (HMC).
    HMC is a Markov chain Monte Carlo (MCMC) algorithm that takes a
    series of gradient-informed steps to produce a Metropolis proposal.
    Mathematical details and derivations can be found in [Neal (2011)].

    The adaptation scheme for this class is `tfp.mcmc.DualAveragingStepSizeAdaptation`
    which is the dual averaging policy that uses a noisy step size for exploration,
    while averaging over tuning steps to provide a smoothed estimate of an optimal value.
    It is based on [section 3.2 of Hoffman and Gelman (2013)], which modifies the
    [stochastic convex optimization scheme of Nesterov (2009)].

    More about the implementation of the HMC:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/DualAveragingStepSizeAdaptation]

    Default values for HMC kernel are:
        {
            step_size:0.1,
            num_leapfrog_steps:3
        }

    Default stat_names:
        ["mean_tree_accept"]

    Default trace_fn:
        Leave only `log_accept_ratio` and calculate deterministic values
    """

    # TODO: provide ref links for papers

    _name = "hmc"
    _grad = True
    _adaptation = mcmc.DualAveragingStepSizeAdaptation
    _kernel = mcmc.HamiltonianMonteCarlo

    default_kernel_kwargs: dict = {"step_size": 0.1, "num_leapfrog_steps": 3}
    default_adapter_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = {"mean_tree_accept"}

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        return (pkr.inner_results.log_accept_ratio,) + tuple(
            self.deterministics_callback(*current_state)
        )


@register_sampler
class HMCSimple(HMC):
    """
    Sampler to run one_step of Hamiltonian Monte Carlo (HMC).
    HMC is a Markov chain Monte Carlo (MCMC) algorithm that takes a
    series of gradient-informed steps to produce a Metropolis proposal.
    Mathematical details and derivations can be found in [Neal (2011)].

    The adaptation scheme for this class is `tfp.mcmc.SimpleStepSizeAdaptation`
    which multiplicatively increases or decreases the step_size of the inner kernel
    based on the value of log_accept_prob. It is based on [equation 19 of Andrieu and Thoms (2008)].


    More about the implementation of the HMC:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/SimpleStepSizeAdaptation]
    """

    _name = "hmc_simple"
    _adaptation = mcmc.SimpleStepSizeAdaptation


@register_sampler
class NUTS(_BaseSampler):
    """
    Sampler to run one_step of No U-Turn Sampler (NUTS).
    NUTS is an adaptive variant of the Hamiltonian Monte
    Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
    the curvature of the target density. Conceptually, one proposal consists of
    reversibly evolving a trajectory through the sample space, continuing until
    that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
    implements one random NUTS step from a given `current_state`.
    Mathematical details and derivations can be found in
    [Hoffman, Gelman (2011)] and [Betancourt (2018)].

    The adaptation scheme for this class is `tfp.mcmc.DualAveragingStepSizeAdaptation`
    which is the dual averaging policy that uses a noisy step size for exploration,
    while averaging over tuning steps to provide a smoothed estimate of an optimal value.
    It is based on [section 3.2 of Hoffman and Gelman (2013)], which modifies the
    [stochastic convex optimization scheme of Nesterov (2009)].

    More about the implementation of the NUTS:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/DualAveragingStepSizeAdaptation]

    Default arguments for NUTS kernel are:
        {step_size=0.1}

    Default arguments for adaptation scheme:
        {
            num_adaptation_steps: 100,
            step_size_getter_fn: lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn: lambda pkr: pkr.log_accept_ratio,
            step_size_setter_fn: lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
        }

    Default `stat_names` values:
        ["lp", "tree_size", "diverging", "energy", "mean_tree_accept"]

    Default `trace_fn` returns:
        (
            inner_results.target_log_prob,
            inner_results.leapfrogs_taken,
            inner_results.has_divergence,
            inner_results.energy,
            inner_results.log_accept_ratio,
            *deterministic_values,
        )

    """

    _name = "nuts"
    _grad = True
    _adaptation = mcmc.DualAveragingStepSizeAdaptation
    _kernel = mcmc.NoUTurnSampler

    # we set default kwargs to support previous sampling logic
    # optimal values can be modified in future
    default_adapter_kwargs: dict = {
        "num_adaptation_steps": 100,
        "step_size_getter_fn": lambda pkr: pkr.step_size,
        "log_accept_prob_getter_fn": lambda pkr: pkr.log_accept_ratio,
        "step_size_setter_fn": lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
    }
    default_kernel_kwargs: dict = {"step_size": 0.1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = ["lp", "tree_size", "diverging", "energy", "mean_tree_accept"]

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        return (
            pkr.inner_results.target_log_prob,
            pkr.inner_results.leapfrogs_taken,
            pkr.inner_results.has_divergence,
            pkr.inner_results.energy,
            pkr.inner_results.log_accept_ratio,
        ) + tuple(self.deterministics_callback(*current_state))


@register_sampler
class NUTSSimple(NUTS):
    """
    Sampler to run one_step of No U-Turn Sampler (NUTS).
    NUTS is an adaptive variant of the Hamiltonian Monte
    Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
    the curvature of the target density. Conceptually, one proposal consists of
    reversibly evolving a trajectory through the sample space, continuing until
    that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
    implements one random NUTS step from a given `current_state`.
    Mathematical details and derivations can be found in
    [Hoffman, Gelman (2011)] and [Betancourt (2018)].

    The adaptation scheme for this class is `tfp.mcmc.SimpleStepSizeAdaptation`
    which multiplicatively increases or decreases the step_size of the inner kernel
    based on the value of log_accept_prob. It is based on [equation 19 of Andrieu and Thoms (2008)].

    More about the implementation of the HMC:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler]

    More about the implementation of the adaptation:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/SimpleStepSizeAdaptation]
    """

    _name = "nuts_simple"
    _adaptation = mcmc.SimpleStepSizeAdaptation


@register_sampler
class RandomWalkM(_BaseSampler):
    """
    Sampler to run one_step of Random Walk Metropolis (RWM).
    RWM is a gradient-free Markov chain Monte Carlo (MCMC)
    algorithm. The algorithm involves a proposal generating
    step proposal_state = current_state + perturb by a random perturbation,
    followed by Metropolis-Hastings accept/reject step. For more details see
    Section 2.1 of Roberts and Rosenthal (2004)

    More about the implementation of the RWM:
        [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/RandomWalkMetropolis]

    Default `stat_names` values:
        ["mean_accept"]

    Default `trace_fn` returns:
        (
            log_accept_ratio,
            *deterministic_values,
        )

    """

    _name = "rwm"
    _adaptation = None
    _kernel = mcmc.RandomWalkMetropolis
    _grad = False

    default_kernel_kwargs: dict = {}
    default_adapter_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = ["mean_accept"]

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        return (pkr.log_accept_ratio,) + tuple(self.deterministics_callback(*current_state))


@register_sampler
class CompoundStep(_BaseSampler):
    """
    The basic implementation of the compound step

    Default `stat_names` values:
        ["compound_results"]

    Default `trace_fn` returns:
        (
            *,
            *deterministic_values,
        )
    """

    _name = "compound"
    _adaptation = None
    _kernel = _CompoundStepTF
    _grad = False

    default_adapter_kwargs: dict = {}
    default_kernel_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_names = ["compound_results"]

    def trace_fn(self, current_state: flow.SamplingState, pkr: Union[tf.Tensor, Any]):
        # TODO: I'm not sure if adding python loop here is the best idea
        # so I'm leaving all the results in trace and discarding it later
        return (pkr,) + tuple(self.deterministics_callback(*current_state))

    @staticmethod
    def _convert_sampler_methods(sampler_methods):
        if not sampler_methods:
            return {}
        sampler_methods_dict = {}
        for sampler_item in sampler_methods:
            # user can pass tuple of lenght=2 or lenght=3
            # we ned to check that.
            if len(sampler_item) not in [2, 3]:
                raise ValueError(
                    "You need to provide `sampler_methods` as the tuple of \
                    the length in [2, 3]. If additional kwargs for kernel \
                    are provided then the lenght of tuple is equal to 3."
                )
            if len(sampler_item) < 3:
                var, sampler = sampler_item
                kwargs = {}
            else:
                var, sampler, kwargs = sampler_item
            if isinstance(var, (list, tuple)):
                for var_ in var:
                    sampler_methods_dict[var_] = (sampler, kwargs)
            else:
                sampler_methods_dict[var] = (sampler, kwargs)
        return sampler_methods_dict

    def _merge_samplers(self, make_kernel_fn, part_kernel_kwargs):
        num_vars = len(make_kernel_fn)
        parents = list(range(num_vars))
        kernels = []

        # Also can be done in O(nlogn) with sorting (maybe more clear solution)
        # With dsu we have explicitly calculated parent indices that are used
        # to sort variables later in logp function

        def cmp_(item1, item2):
            # hack to separetely compare proposal function and
            # the rest of the kwargs of the sampler kernel
            key = "new_state_fn"
            if (key in item1 and key in item2) or (key not in item1 and key not in item2):
                # compare class instances instea of _fn for `new_state_fn`
                if key in item1:
                    if item1[key].__self__ != item2[key].__self__:
                        return False
                    value1, value2 = item1.pop(key), item2.pop(key)
                    flag = item1 == item2
                    item1[key], item2[key] = value1, value2
                else:
                    flag = item1 == item2
                return flag
            else:
                return False

        # DSU ops
        def get_set(p):
            return p if parents[p] == p else get_set(parents[p])

        def union_set(p1, p2):
            p1, p2 = get_set(p1), get_set(p2)
            if p1 != p2:
                parents[max(p1, p2)] = min(p1, p2)

        # merge sets, DSU
        for (i, j) in itertools.combinations(range(num_vars), 2):
            # For the sampler to be the same we are comparing
            # both classes of the chosen sampler and the key
            # arguments.
            if make_kernel_fn[i] == make_kernel_fn[j] and cmp_(
                part_kernel_kwargs[i], part_kernel_kwargs[j]
            ):
                union_set(i, j)
        # assign kernels based on unique sets

        used_p = {}
        for i, p in enumerate(parents):
            if p not in used_p:
                kernels.append((make_kernel_fn[i], part_kernel_kwargs[i]))
                used_p[p] = True

        # calculate independent set lengths
        parent_used, set_lengths = {}, []
        for p in parents:
            if p in parent_used:
                set_lengths[parent_used[p]] += 1
            else:
                parent_used[p] = len(set_lengths)
                set_lengths.append(1)
        self.parent_inds = sorted(range(len(parents)), key=lambda k: parents[k])
        return kernels, set_lengths

    def _assign_default_methods(
        self,
        *,
        sampler_methods: Optional[List] = None,
        state: Optional[flow.SamplingState] = None,
        observed: Optional[dict] = None,
    ):
        converted_sampler_methods: List = CompoundStep._convert_sampler_methods(sampler_methods)

        (_, state, _, _, continuous_distrs, discrete_distrs) = initialize_state(
            self.model, observed=observed, state=state
        )
        init = state.all_unobserved_values
        init_state = list(init.values())
        init_keys = list(init.keys())

        # assignd samplers for free variables
        make_kernel_fn: list = []
        # user passed kwargs for each sampler in `make_kernel_fn`
        part_kernel_kwargs: list = []
        # keep the list for proposal func names
        func_names: list = []

        for i, state_part in enumerate(init_state):
            untrs_var, unscoped_tr_var = scope_remove_transformed_part_if_required(
                init_keys[i], state.transformed_values
            )
            # get the distribution for the random variable name

            distr = continuous_distrs.get(untrs_var, None)
            if distr is None:
                distr = discrete_distrs[untrs_var]

            # get custom `new_state_fn` for the distribution
            func = distr._default_new_state_part

            # simplest way of assigning sampling methods
            # if the sampler_methods was passed and if a var is provided
            # then the var will be assigned to the given sampler
            # but will also be checked if the sampler supports the distr

            # 1. If sampler is provided by the user, we create new sampler
            #    and add to `make_kernel_fn`
            # 2. If the distribution has `new_state_fn` then the new sampler
            #    should be create also. Because sampler is initialized with
            #    the `new_state_fn` argument.
            if unscoped_tr_var in converted_sampler_methods:
                sampler, kwargs = converted_sampler_methods[unscoped_tr_var]

                # check for the sampler able to sampler from the distribution
                if not distr._grad_support and sampler._grad:
                    raise ValueError(
                        "The `{}` doesn't support gradient, please provide an appropriate sampler method".format(
                            unscoped_tr_var
                        )
                    )

                # add sampler to the dict
                make_kernel_fn.append(sampler)
                part_kernel_kwargs.append({})
                # update with user provided kwargs
                part_kernel_kwargs[-1].update(kwargs)
                # if proposal function is provided then replace
                func = part_kernel_kwargs[-1].get("new_state_fn", func)
                # add the default `new_state_fn` for the distr
                # `new_state_fn` is supported for only RandomWalkMetropolis transition
                # kernel.
                if func and sampler._name == "rwm":
                    part_kernel_kwargs[-1]["new_state_fn"] = partial(func)()
            elif callable(func):
                # If distribution has defined `new_state_fn` attribute then we need
                # to assign `RandomWalkMetropolis` transition kernel
                make_kernel_fn.append(RandomWalkM)
                part_kernel_kwargs.append({})
                part_kernel_kwargs[-1]["new_state_fn"] = partial(func)()
            else:
                # by default if user didn't not provide any sampler
                # we choose NUTS for the variable with gradient and
                # RWM for the variable without the gradient
                sampler = NUTS if distr._grad_support else RandomWalkM
                make_kernel_fn.append(sampler)
                part_kernel_kwargs.append({})
                # _log.info("Auto-assigning NUTS sampler...")
            # save proposal func names
            func_names.append(func._name if func else "default")

        # `make_kernel_fn` contains (len(state)) sampler methods, this could lead
        # to more overhed when we are iterating at each call of `one_step` in the
        # compound step kernel. For that we need to merge some of the samplers.
        kernels, set_lengths = self._merge_samplers(make_kernel_fn, part_kernel_kwargs)
        # log variable sampler mapping
        CompoundStep._log_variables(init_keys, kernels, set_lengths, self.parent_inds, func_names)
        # save to use late for compound kernel init
        self.kernel_kwargs["compound_samplers"] = kernels
        self.kernel_kwargs["compound_set_lengths"] = set_lengths

    def __call__(self, *args, **kwargs):
        return self._sample(*args, is_compound=True, **kwargs)

    @staticmethod
    def _log_variables(var_keys, kernel_kwargs, set_lengths, parent_inds, func_names):
        var_keys = [var_keys[i] for i in parent_inds]
        func_names = [func_names[i] for i in parent_inds]
        log_output = ""
        curr_indx = 0
        for i, (kernel_kwargsi, set_leni) in enumerate(zip(kernel_kwargs, set_lengths)):
            kernel, kwargs = kernel_kwargsi
            vars_ = var_keys[curr_indx : curr_indx + set_leni]
            log_output += ("\n" if i > 0 else "") + " -- {}[vars={}, proposal_function={}]".format(
                kernel._name,
                [item.split("/")[1] for item in vars_],
                (func_names[curr_indx]),
            )
            curr_indx += set_leni
        _log.info(log_output)


def build_logp_and_deterministic_functions(
    model,
    num_chains: Optional[int] = None,
    observed: Optional[dict] = None,
    state: Optional[flow.SamplingState] = None,
    collect_reduced_log_prob: bool = True,
    parent_inds: Optional[List] = None,
):
    if not isinstance(model, Model):
        raise TypeError(
            "`sample` function only supports `pymc4.Model` objects, but you've passed `{}`".format(
                type(model)
            )
        )
    if state is not None and observed is not None:
        raise ValueError("Can't use both `state` and `observed` arguments")

    state, deterministic_names = initialize_sampling_state(model, observed=observed, state=state)

    if not state.all_unobserved_values:
        raise ValueError(
            f"Can not calculate a log probability: the model {model.name or ''} has no unobserved values."
        )

    observed_var = state.observed_values
    unobserved_keys, unobserved_values = zip(*state.all_unobserved_values.items())

    if parent_inds:
        unobserved_keys = [unobserved_keys[i] for i in parent_inds]
        unobserved_values = [unobserved_values[i] for i in parent_inds]

    if collect_reduced_log_prob:

        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs, observed_values=observed)
            _, st = flow.evaluate_model_transformed(model, state=st)
            return st.collect_log_prob()

    else:
        # When we use manual batching, we need to manually tile the chains axis
        # to the left of the observed tensors
        if num_chains is not None:
            obs = state.observed_values
            if observed is not None:
                obs.update(observed)
            else:
                observed = obs
            for k, o in obs.items():
                o = tf.convert_to_tensor(o)
                o = tf.tile(o[None, ...], [num_chains] + [1] * o.ndim)
                observed[k] = o

        @tf.function(autograph=False)
        def logpfn(*values, **kwargs):
            if kwargs and values:
                raise TypeError("Either list state should be passed or a dict one")
            elif values:
                kwargs = dict(zip(unobserved_keys, values))
            st = flow.SamplingState.from_values(kwargs, observed_values=observed)
            _, st = flow.evaluate_model_transformed(model, state=st)
            return st.collect_unreduced_log_prob()

    @tf.function(autograph=False)
    def deterministics_callback(*values, **kwargs):
        if kwargs and values:
            raise TypeError("Either list state should be passed or a dict one")
        elif values:
            kwargs = dict(zip(unobserved_keys, values))
        st = flow.SamplingState.from_values(kwargs, observed_values=observed_var)
        _, st = flow.evaluate_model_transformed(model, state=st)
        for transformed_name in st.transformed_values:
            untransformed_name = NameParts.from_name(transformed_name).full_untransformed_name
            st.deterministics_values[untransformed_name] = st.untransformed_values.pop(
                untransformed_name
            )
        return st.deterministics_values.values()

    return (
        logpfn,
        dict(state.all_unobserved_values),
        deterministics_callback,
        deterministic_names,
        state,
    )


def vectorize_logp_function(logpfn):
    # TODO: vectorize with dict
    def vectorized_logpfn(*state):
        return tf.vectorized_map(lambda mini_state: logpfn(*mini_state), state)

    return vectorized_logpfn


def tile_init(init, num_repeats):
    return [tf.tile(tf.expand_dims(tens, 0), [num_repeats] + [1] * tens.ndim) for tens in init]


def calculate_log_likelihood(
    model: Model, posterior: Dict[str, tf.Tensor], sampling_state: flow.SamplingState
) -> Dict[str, tf.Tensor]:
    """Compute log likelihood by vectorizing chains and draws."""

    def extract_log_likelihood(values, observed_rv):
        st = flow.SamplingState.from_values(values, observed_values=sampling_state.observed_values)
        _, st = flow.evaluate_model_transformed(model, state=st)
        try:
            dist = st.continuous_distributions[observed_rv]
        except KeyError:
            dist = st.discrete_distributions[observed_rv]
        return dist.log_prob(dist.model_info["observed"])

    # First vectorizing chains and then draws, but ultimately, draws and chains get
    # swapped for mcmc trace while passing to `trace_to_arviz`.
    log_likelihood_dict = dict()
    for observed_rv in sampling_state.observed_values:
        extract_log_likelihood = partial(extract_log_likelihood, observed_rv=observed_rv)
        vectorized_chains = vectorize_logp_function(extract_log_likelihood)
        vectorized_draws = vectorize_logp_function(vectorized_chains)
        log_likelihood_dict[observed_rv] = vectorized_draws(posterior)
    return log_likelihood_dict
