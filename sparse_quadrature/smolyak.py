from collections import namedtuple
from itertools import combinations_with_replacement, permutations, chain
import numpy as np
from utils.numbas import isin_nd
OneDNodesAndWeights = namedtuple("OneDNodesAndWeights", ['level', 'roots', 'weights'])


def isin_nd(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
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
    >>> a = np.array([1,0])
    >>> A = np.array([[0,0],[1,0]])
    >>> isin_nd(A,a)
    array([False,  True])
    >>> B = np.array([[0, 1], [0,0], [1, 3]])
    >>> isin_nd(B, A)
    array([False,  True, False])

    """
    return (array_a[:, None] == array_b).all(-1).any(-1)


def one_d_difference_grid(one_d_nodes_and_weights_list: list) \
        -> list[OneDNodesAndWeights]:
    """
    Compute the difference grid nodes and their weights.
    The resulted grids are might have duplicated points.

    Parameters
    ----------
    one_d_nodes_and_weights_list

    Returns
    -------

    """
    difference_grids = []
    isin = np.isin

    for i in range(len(one_d_nodes_and_weights_list)):
        ondrw = one_d_nodes_and_weights_list[i]
        if ondrw.level == 1:
            difference_grids.append(one_d_nodes_and_weights_list[i])
        else:
            ondrw_m_1 = one_d_nodes_and_weights_list[i - 1]
            similar_roots_flags = isin(ondrw.roots, ondrw_m_1.roots)
            if np.any(similar_roots_flags):
                similar_roots = ondrw.roots[similar_roots_flags]

                # although the root positions are the same, but the weight are different
                similar_weights = ondrw.weights[similar_roots_flags] - ondrw_m_1.weights[
                    ondrw_m_1.roots == similar_roots]
            else:
                similar_roots = np.array([])
                similar_weights = np.array([])

            current_roots = ondrw.roots[~ similar_roots_flags]
            current_weights = ondrw.weights[~ similar_roots_flags]

            similar_roots_flags = isin(ondrw_m_1.roots, ondrw.roots)
            prev_roots = ondrw_m_1.roots[~ similar_roots_flags]
            prev_weights = - ondrw_m_1.weights[~ similar_roots_flags]  # this is minus

            delta_weights = np.concatenate((current_weights, prev_weights, similar_weights))
            delta_roots = np.concatenate((current_roots, prev_roots, similar_roots))

            difference_grids.append(OneDNodesAndWeights(ondrw.level, delta_roots, delta_weights))

    return difference_grids


def smolyak_indices(dim: int, max_level: int) -> np.array:
    """

    Parameters
    ----------
    dim
    max_level

    Returns
    -------

    """
    # Need to capture up to value mu + 1 so in python need mu+2
    possible_values = range(1, max_level + 1)

    # find all (i1, i2, ... id) such that their sum is in range
    # we want; this will cut down on later iterations
    raw_indices = np.array(list(combinations_with_replacement(possible_values, dim)))

    # limit the indices only to those satisfies the bound
    poss_indices = raw_indices[np.sum(raw_indices, axis=1) < dim + max_level]

    # use set to remove duplicate entries.
    true_inds = [[el for el in set(permutations(val))] for val in poss_indices]

    return np.array(list(chain(*true_inds)))


def get_raw_grids(smol_indices: np.ndarray, difference_grids: list[OneDNodesAndWeights]) -> tuple[
    np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    smol_indices
    difference_grids

    Returns
    -------

    """
    dim = smol_indices.shape[1]

    raw_grid_points = []
    raw_weights = []
    for indices in smol_indices:
        one_d_grids = []
        one_d_weights = []
        number_of_elements = 1
        for index in indices:
            roots = difference_grids[index - 1].roots
            weights = difference_grids[index - 1].weights
            one_d_grids.append(roots)
            one_d_weights.append(weights)
            number_of_elements *= roots.shape[0]

        grids = np.stack(np.meshgrid(*one_d_grids), axis=-1)
        grids = grids.reshape((number_of_elements, dim))
        nd_weights = np.stack(np.meshgrid(*one_d_weights), axis=-1)
        nd_weights = np.prod(nd_weights.reshape((number_of_elements, dim)), axis=-1)

        raw_grid_points.append(grids)
        raw_weights.append(nd_weights)

    raw_grid_points = np.stack(list(chain(*raw_grid_points)))
    raw_weights = np.stack(list(chain(*raw_weights)))
    return raw_grid_points, raw_weights


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
    unique_grid_points = np.unique(raw_grid_points, axis=0)
    unique_weights = np.zeros((unique_grid_points.shape[0]))
    for i in range(unique_grid_points.shape[0]):
        a_point = unique_grid_points[i]
        flags = isin_nd(raw_grid_points, a_point)
        unique_weights[i] = np.sum(raw_weights[flags])

    if weight_cut_off:
        # remove those that has zero weights, or close to zero
        flags = np.isclose(np.abs(unique_weights), 0, rtol=weight_cut_off)
        unique_weights = unique_weights[~ flags]
        unique_grid_points = unique_grid_points[~ flags]

    return smol_indices, unique_grid_points, unique_weights
