import numpy as np
import collections
import tensorflow as tf


def make_shared_vectorized_input(rvs, test_values):
    dim = 0
    for shape in rvs.items():
        dim += np.prod(shape.as_list())
    init = np.empty(dim)
    j = 0
    for name, shape in rvs.items():
        d = np.prod(shape.as_list())
        init[j:d] = np.asarray(test_values[name]).flatten()
        j += d
    vec_shape = (dim, )
    with tf.variable_scope('vectorize'):
        vec = tf.get_variable('vec_state', vec_shape, initializer=tf.constant_initializer(init))
        j = 0
        mapping = collections.OrderedDict()
        with tf.name_scope('slice'):
            for name, shape in rvs.items():
                d = np.prod(shape.as_list())
                mapping[name] = tf.reshape(vec[j:d], shape, name=name)
                j += d
    return vec, mapping


# from Theano codebase
def stack_search(start, expand, mode='bfs', build_inv=False):
    """
    Search through a graph, either breadth- or depth-first.

    Parameters
    ----------
    start : deque
        Search from these nodes.
    expand : callable
        When we get to a node, add expand(node) to the list of nodes to visit.
        This function should return a list, or None.
    mode : string
        'bfs' or 'dfs' for breath first search or depth first search.

    Returns
    -------
    list of `Variable` or `Apply` instances (depends on `expend`)
        The list of nodes in order of traversal.

    Notes
    -----
    A node will appear at most once in the return value, even if it
    appears multiple times in the start parameter.

    :postcondition: every element of start is transferred to the returned list.
    :postcondition: start is empty.

    """

    if mode not in ('bfs', 'dfs'):
        raise ValueError('mode should be bfs or dfs', mode)
    rval_set = set()
    rval_list = list()
    if mode == 'bfs':
        start_pop = start.popleft
    else:
        start_pop = start.pop
    expand_inv = {}  # var: clients
    while start:
        l = start_pop()
        if id(l) not in rval_set:
            rval_list.append(l)
            rval_set.add(id(l))
            expand_l = expand(l)
            if expand_l:
                if build_inv:
                    for r in expand_l:
                        expand_inv.setdefault(r, []).append(l)
                start.extend(expand_l)
    assert len(rval_list) == len(rval_set)
    if build_inv:
        return rval_list, expand_inv
    return rval_list


# from Theano codebase
def ancestors(variable_list, blockers=None):
    """
    Return the variables that contribute to those in variable_list (inclusive).

    Parameters
    ----------
    variable_list : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.

    Returns
    -------
    list of `Variable` instances
        All input nodes, in the order found by a left-recursive depth-first
        search started at the nodes in `variable_list`.
    """
    def expand(r):
        if r.owner and (not blockers or r not in blockers):
            return reversed(r.owner.inputs)
    dfs_variables = stack_search(collections.deque(variable_list), expand, 'dfs')
    return dfs_variables


# from Theano codebase
def inputs(variable_list, blockers=None):
    """
    Return the inputs required to compute the given Variables.

    Parameters
    ----------
    variable_list : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.

    Returns
    -------
    list of `Variable` instances
        Input nodes with no owner, in the order found by a left-recursive
        depth-first search started at the nodes in `variable_list`.

    """
    from ..model import InputDistribution
    vlist = ancestors(variable_list, blockers)
    rval = [r for r in vlist if hasattr(r, 'distribution') and
            isinstance(r.distribution, InputDistribution)]
    return rval
