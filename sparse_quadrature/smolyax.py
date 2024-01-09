from collections import namedtuple
from itertools import combinations_with_replacement, permutations, chain
import numpy as onp
import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial
from utils.hashable_array import HashableArrayWrapper

OneDNodesAndWeightsX = namedtuple("OneDNodesAndWeightsX", ['level', 'roots', 'weights'])


def nwx_equal(nw_a: OneDNodesAndWeightsX, nw_b: OneDNodesAndWeightsX):
    equal_level = nw_a.level == nw_b.level
    equal_roots = jnp.all(jnp.equal(nw_a.roots, nw_b.roots))
    equal_weights = jnp.all(jnp.equal(nw_a.weights, nw_b.roots))

    return equal_weights and equal_roots and equal_level


def nwx_hash(nw: OneDNodesAndWeightsX):
    return nw.level + int(jnp.sum(nw.roots) + jnp.sum(nw.weights))


class HashableOneDNWXWrapper:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return nwx_hash(self.val)

    def __eq__(self, other):
        return isinstance(other, HashableOneDNWXWrapper) and nwx_equal(self.val, other.val)


def nwx_jit(fun, static_nwx_argnums=()):
    @partial(jit, static_argnums=static_nwx_argnums)
    def callee(*args):
        args = list(args)
        for i in static_nwx_argnums:
            # print("Executed")
            args[i] = args[i].val
        return fun(*args)

    def caller(*args):
        args = list(args)
        for i in static_nwx_argnums:
            args[i] = HashableOneDNWXWrapper(args[i])

        return callee(*args)

    return caller


def combined_jit(fun, static_argnums=()):
    @partial(jit, static_argnums=static_argnums)
    def callee(*args):
        args = list(args)
        for i in static_argnums:
            if not isinstance(args[i], tuple):
                args[i] = args[i].val
            else:
                temp = []
                for a in args[i]:
                    temp.append(a.val)
                args[i] = tuple(temp)

        return fun(*args)

    def caller(*args):
        args = list(args)
        for i in static_argnums:

            if isinstance(args[i], tuple):
                print("Executed here. Type of args[{}][0] is {}".format(i, type(args[i][0])))
                temp = args[i]
                nwxl = []
                for nwx in temp:
                    nwxl.append(HashableOneDNWXWrapper(nwx))
                args[i] = tuple(nwxl)
            elif isinstance(args[i], jnp.ndarray):
                args[i] = HashableArrayWrapper(args[i])

        return callee(*args)

    return caller


@jit
def isin_nd(array_a: jnp.ndarray, array_b: jnp.ndarray) -> jnp.ndarray:
    """
    This works like `np.isin` but in multidimensional settings.

    Parameters
    ----------
    array_a
    array_b

    Returns
    -------

    Examples
    --------
    >>> a = jnp.array([1,0])
    >>> A = jnp.array([[0,0],[1,0]])
    >>> isin_nd(A,a)
    DeviceArray([False,  True], dtype=bool)
    >>> B = jnp.array([[0, 1], [0,0], [1, 3]])
    >>> isin_nd(B, A)
    DeviceArray([False,  True, False], dtype=bool)
    """
    return (array_a[:, None] == array_b).all(-1).any(-1)

@partial(combined_jit, static_argnums=(0,))
def one_d_difference_grid(one_d_nodes_and_weights_list: tuple) \
        -> list[OneDNodesAndWeightsX]:
    """
    Compute the difference grid nodes and their weights.
    The resulted grids are might have duplicated points.

    Parameters
    ----------
    one_d_nodes_and_weights_list

    Returns
    -------

    Examples
    --------
    >>> from sparse_quadrature.patterson import one_d_patterson_nodes_up_to_a_level
    >>> from sparse_quadrature.patterson import one_d_patterson_nodes_up_to_a_levelx
    >>> import sparse_quadrature.smolyak as smol
    >>> import numpy as np
    >>> level = 2
    >>> n_w_lists = one_d_patterson_nodes_up_to_a_level(level)
    >>> n_w_listsx = one_d_patterson_nodes_up_to_a_levelx(level)
    >>> od_dg = smol.one_d_difference_grid(n_w_lists)
    >>> od_dg_x = one_d_difference_grid(n_w_listsx)
    >>> len(od_dg) == len(od_dg_x)
    True
    """
    difference_grids = []

    for i in range(len(one_d_nodes_and_weights_list)):
        odnrw = one_d_nodes_and_weights_list[i]
        if odnrw.level == 1:
            difference_grids.append(one_d_nodes_and_weights_list[i])
        else:
            odnrw_m_1 = one_d_nodes_and_weights_list[i - 1]
            similar_roots_flags = onp.isin(odnrw.roots, odnrw_m_1.roots)
            if onp.any(similar_roots_flags):
                similar_roots = odnrw.roots[similar_roots_flags]

                # although the root positions are the same, but the weight are different
                similar_weights = odnrw.weights[similar_roots_flags] - odnrw_m_1.weights[
                    odnrw_m_1.roots == similar_roots]
            else:
                similar_roots = jnp.array([])
                similar_weights = jnp.array([])

            current_roots = odnrw.roots[~ similar_roots_flags]
            current_weights = odnrw.weights[~ similar_roots_flags]

            similar_roots_flags = jnp.isin(odnrw_m_1.roots, odnrw.roots)
            prev_roots = odnrw_m_1.roots[~ similar_roots_flags]
            prev_weights = - odnrw_m_1.weights[~ similar_roots_flags]  # this is minus

            delta_weights = jnp.concatenate((current_weights, prev_weights, similar_weights))
            delta_roots = jnp.concatenate((current_roots, prev_roots, similar_roots))

            difference_grids.append(OneDNodesAndWeightsX(odnrw.level, delta_roots, delta_weights))

    return difference_grids


# @partial(jit, static_argnums=(0,1,))
def smolyak_indices(dim: int, max_level: int) -> jnp.array:
    """

    Parameters
    ----------
    dim
    max_level

    Returns
    -------

    Examples
    --------

    >>> import sparse_quadrature.smolyak as smol
    >>> import numpy as np
    >>> dim = 2
    >>> level = 2
    >>> raw_i = smol.smolyak_indices(dim,level)
    >>> rawx_i = smolyak_indices(dim,level)
    >>> onp.all(onp.equal(onp.array(rawx_i), raw_i))
    True

    """
    # Need to capture up to value mu + 1 so in python need mu+2
    possible_values = range(1, max_level + 1)

    # find all (i1, i2, ... id) such that their sum is in range
    # we want; this will cut down on later iterations
    raw_indices = jnp.array(list(combinations_with_replacement(possible_values, dim)))

    # limit the indices only to those satisfies the bound
    poss_indices = raw_indices[jnp.sum(raw_indices, axis=1) < dim + max_level]

    # use set to remove duplicate entries.
    true_inds = [[el for el in set(permutations(val))] for val in onp.array(poss_indices)]

    return jnp.array(list(chain(*true_inds)))


@partial(combined_jit, static_argnums=[0, 1])
def get_raw_grids(smol_indices: jnp.ndarray, difference_grids: list[OneDNodesAndWeightsX]) -> tuple[
    jnp.ndarray, jnp.ndarray]:
    """

    Parameters
    ----------
    smol_indices
    difference_grids

    Returns
    -------

    Examples
    --------
    >>> from sparse_quadrature.patterson import one_d_patterson_nodes_up_to_a_level
    >>> from sparse_quadrature.patterson import one_d_patterson_nodes_up_to_a_levelx
    >>> import sparse_quadrature.smolyak as smol
    >>> import numpy as np
    >>> dim = 2
    >>> level = 2
    >>> n_w_lists = one_d_patterson_nodes_up_to_a_level(level)
    >>> n_w_listsx = one_d_patterson_nodes_up_to_a_levelx(level)
    >>> od_dg = smol.one_d_difference_grid(n_w_lists)
    >>> od_dg_x = one_d_difference_grid(n_w_listsx)
    >>> smids = smol.smolyak_indices(dim,level)
    >>> smidsx = smolyak_indices(dim,level)
    >>> rgp, rgw = smol.get_raw_grids(smids,n_w_lists)
    >>> rgpx, rgwx = smolx.get_raw_grids(smidsx,n_w_listsx)
    >>> rgp.shape == (1,)
    True

    """

    def _scan_inner(carry_, input_):
        index_ = input_
        difference_grids_, number_of_elements_ = carry_
        roots_ = difference_grids_[index_ - 1].roots
        weights_ = difference_grids_[index_ - 1].weights
        number_of_elements_ *= roots_.shape[0]
        return (difference_grids_, number_of_elements_), (roots_, weights_)

    def _scan_outer(carry_, input_):
        indices_ = input_
        dim = indices_.shape[0]
        (_, number_of_elements), (one_d_grids, one_d_weights) = lax.scan(_scan_inner,
                                                                         (difference_grids, 1), indices_)

        grids = jnp.stack(jnp.meshgrid(*one_d_grids), axis=-1)
        grids = grids.reshape((number_of_elements, dim))
        nd_weights = jnp.stack(jnp.meshgrid(*one_d_weights), axis=-1)
        nd_weights = jnp.prod(nd_weights.reshape((number_of_elements, dim)), axis=-1)

        return None, (grids, nd_weights)

    _, (raw_grid_points, raw_weights) = lax.scan(_scan_outer, None, smol_indices)
    return raw_grid_points, raw_weights


@partial(jit, static_argnums=[1, ])
def smolyak_grid(dim: int, one_d_nodes_and_weights_list: list, weight_cut_off: float = None):
    """
    A naive numpy based construction of smolyak grid.
    Probably would be ok for low dimension.


    Parameters
    ----------

    dim
    one_d_nodes_and_weights_list
    weight_cut_off

    Returns
    -------

    """
    difference_grids = one_d_difference_grid(one_d_nodes_and_weights_list)
    max_level = len(difference_grids)
    smol_indices = smolyak_indices(dim, max_level)
    raw_grid_points, raw_weights = get_raw_grids(smol_indices, difference_grids)
    # gather similar points, and accumulate their weights
    unique_grid_points = onp.unique(raw_grid_points, axis=0)
    unique_weights = onp.zeros((unique_grid_points.shape[0]))
    for i in range(unique_grid_points.shape[0]):
        a_point = unique_grid_points[i]
        flags = isin_nd(raw_grid_points, a_point)
        unique_weights[i] = onp.sum(raw_weights[flags])

    if weight_cut_off:
        # remove those that has zero weights, or close to zero
        flags = onp.isclose(onp.abs(unique_weights), 0, rtol=weight_cut_off)
        unique_weights = unique_weights[~ flags]
        unique_grid_points = unique_grid_points[~ flags]

    return smol_indices, unique_grid_points, unique_weights
