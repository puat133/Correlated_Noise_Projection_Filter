from collections import namedtuple
from itertools import combinations_with_replacement, permutations, chain
import numpy as np
import numba as nb
from sparse_quadrature.smolyak import one_d_difference_grid, smolyak_indices
from numba.typed import List as typed_list

OneDNodesAndWeights = namedtuple("OneDNodesAndWeights", ['level', 'roots', 'weights'])


@nb.njit(fastmath=True)
def maxnorm(a: np.ndarray) -> float:
    """
    Calculate maxnorm

    Parameters
    ----------
    a : np.ndarray

    Returns
    -------
    maxnorm : float

    Examples
    --------
    >>> a = np.array([1.,2.,3.])
    >>> maxnorm(a)
    3.0
    """
    return np.max(np.abs(a))


@nb.njit(fastmath=True)
def isclose(a: np.array, b: np.array, rtol: float):
    """

    Check whether two array of the same size is close each other.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : float

    Returns
    -------
    decision : bool

    Examples
    --------
    >>> a = np.array([1,2,3])
    >>> b = np.array([1,2,3+1e-9])
    >>> isclose(a,b,1e-5)
    True

    """
    norm_diff = maxnorm(a - b)
    max_ = max(maxnorm(a), maxnorm(b))
    if max_ > 0:
        result = norm_diff / max_ < rtol
    else:
        result = True
    return result


@nb.vectorize([nb.boolean(nb.float64, nb.float64, nb.float64),
               nb.boolean(nb.float32, nb.float32, nb.float32)])
def isclose_scalar(a: float, b: float, rtol: float) -> bool:
    return np.absolute(a - b) < rtol


@nb.njit(fastmath=True)
def isin_nd(a_array: np.ndarray, b_array: np.ndarray, rtol: float = 1e-8):
    """

    This works like `np.isin` but in multidimensional settings.

    Parameters
    ----------
    a_array
    b_array
    rtol

    Returns
    -------

    Examples
    --------
    >>> a = np.array([1.,0.])
    >>> A = np.array([[0.,0.],[1.,0.]])
    >>> isin_nd(A,a)
    array([False,  True])
    >>> B = np.array([[0., 1.], [0.,0.], [1., 3.]])
    >>> isin_nd(B, A)
    array([False,  True, False])

    """
    result = np.zeros((a_array.shape[0],), dtype=nb.boolean)
    dim = b_array.ndim

    if dim > 1:
        for i in range(a_array.shape[0]):
            a = a_array[i]
            for j in range(b_array.shape[0]):
                if isclose(a, b_array[j], rtol=rtol):
                    result[i] = True
                break

    else:
        for i in range(a_array.shape[0]):
            a = a_array[i]
            if isclose(a, b_array, rtol=rtol):
                result[i] = True
                break
    return result


# @nb.njit()
# def one_d_difference_grid(one_d_nodes_and_weights_list: list) \
#         -> list[OneDNodesAndWeights]:
#     """
#     Compute the difference grid nodes and their weights.
#     The resulted grids are might have duplicated points.
#
#     Parameters
#     ----------
#     one_d_nodes_and_weights_list
#
#     Returns
#     -------
#
#     """
#     difference_grids = typed_list()
#     for i in range(len(one_d_nodes_and_weights_list)):
#         ondrw = one_d_nodes_and_weights_list[i]
#         if ondrw[0] == 1:
#             difference_grids.append((one_d_nodes_and_weights_list[i]))
#         else:
#             ondrw_m_1 = one_d_nodes_and_weights_list[i - 1]
#             similar_roots_flags = isin(ondrw.roots, ondrw_m_1.roots)
#             if np.any(similar_roots_flags):
#                 similar_roots = ondrw[1][similar_roots_flags]
#                 similar_weights = ondrw[2][similar_roots_flags] - ondrw_m_1[2][
#                     ondrw_m_1[1] == similar_roots]
#             else:
#                 similar_roots = np.empty((0,))
#                 similar_weights = np.empty((0,))
#
#             current_roots = ondrw[1][~ similar_roots_flags]
#             current_weights = ondrw[2][~ similar_roots_flags]
#
#             similar_roots_flags = isin(ondrw_m_1[1], ondrw[1])
#             prev_roots = ondrw_m_1[1][~ similar_roots_flags]
#             prev_weights = - ondrw_m_1[2][~ similar_roots_flags]  # this is minus
#
#             delta_weights = np.concatenate((current_weights, prev_weights, similar_weights))
#             delta_roots = np.concatenate((current_roots, prev_roots, similar_roots))
#
#             difference_grids.append((ondrw[0], delta_roots, delta_weights))
#
#     return difference_grids


# def smolyak_indices(dim: int, max_level: int) -> np.array:
#     """
#
#     Parameters
#     ----------
#     dim
#     max_level
#
#     Returns
#     -------
#
#     """
#     # Need to capture up to value mu + 1 so in python need mu+2
#     possible_values = range(1, max_level + 1)
#
#     # find all (i1, i2, ... id) such that their sum is in range
#     # we want; this will cut down on later iterations
#     raw_indices = np.array(list(combinations_with_replacement(possible_values, dim)))
#
#     # limit the indices only to those satisfies the bound
#     poss_indices = raw_indices[np.sum(raw_indices, axis=1) < dim + max_level]
#
#     # use set to remove duplicate entries.
#     true_inds = [[el for el in set(permutations(val))] for val in poss_indices]
#
#     return np.array(list(chain(*true_inds)))

@nb.njit(fastmath=True)
def make_raw_grid(smol_indices, difference_grids, dim):
    raw_grid_points = np.empty((0, dim))
    raw_weights = np.empty((0,))
    for indices in smol_indices:
        one_d_grids = []
        one_d_weights = []
        number_of_elements = 1
        for index in indices:
            roots = difference_grids[index - 1][1]
            weights = difference_grids[index - 1][2]
            one_d_grids.append(roots)
            one_d_weights.append(weights)
            number_of_elements *= roots.shape[0]

        grids = np.stack(np.meshgrid(*one_d_grids), axis=-1)
        grids = grids.reshape((number_of_elements, dim))
        nd_weights = np.stack(np.meshgrid(*one_d_weights), axis=-1)
        nd_weights = np.prod(nd_weights.reshape((number_of_elements, dim)), axis=-1)

        raw_grid_points = np.vstack([raw_grid_points, grids])
        raw_weights = np.concatenate([raw_weights, nd_weights])

    return raw_grid_points, raw_weights


@nb.njit(fastmath=True)
def make_weight_unique(unique_grid_points: np.ndarray, raw_grid_points: np.ndarray, raw_weights: np.ndarray):
    # gather similar points, and accumulate their weights

    unique_weights = np.zeros((unique_grid_points.shape[0]))
    for i in range(unique_grid_points.shape[0]):
        a_point = np.atleast_2d(unique_grid_points[i])
        flags = isin(raw_grid_points, a_point)
        unique_weights[i] = np.sum(raw_weights[flags])

    return unique_weights


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
    raw_grid_points, raw_weights = make_raw_grid(smol_indices, difference_grids, dim)
    unique_grid_points = np.unique(raw_grid_points, axis=0)
    unique_weights = make_weight_unique(unique_grid_points, raw_grid_points, raw_weights)

    if weight_cut_off:
        # remove those that has zero weights, or close to zero
        flags = np.isclose(np.abs(unique_weights), 0, rtol=weight_cut_off)
        unique_weights = unique_weights[~ flags]
        unique_grid_points = unique_grid_points[~ flags]

    return smol_indices, unique_grid_points, unique_weights
