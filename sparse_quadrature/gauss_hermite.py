import numpy as np
from sparse_quadrature.smolyak import OneDNodesAndWeights, smolyak_grid
from scipy.special import roots_hermite


def one_d_gauss_hermite_nodes_up_to_a_level(a_level: int) -> list[OneDNodesAndWeights]:
    """

    Parameters
    ----------
    a_level : int
        a level

    Returns
    -------
    one_d_nodes_and_weight : list
        list of the form (leve, roots, weights)

    References
    ---------
    build based on https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermite.html
    """

    one_d_nodes_and_weights_list = []

    # This is looks like the one in AIAA paper
    # polynomial order will be 1, 3, 7, ...
    # polynomial_order_gap = np.concatenate(([0], np.cumsum(np.power(2, np.arange(1, a_level + 2)))))
    # polynomial_order_gap = np.arange(a_level + 1)
    polynomial_orders = np.cumsum(np.power(2, np.arange(a_level + 1)))

    for i in range(a_level + 1):
        polynomial_order = polynomial_orders[i]
        roots, weights = roots_hermite(polynomial_order)
        one_d_nodes_and_weights_list.append(OneDNodesAndWeights(i + 1, roots, weights))

    return one_d_nodes_and_weights_list


def sparse_gauss_hermite_quadrature(dim: int, level: int, weight_cut_off: float = None):
    """

    Parameters
    ----------

    dim
    level
    weight_cut_off

    Returns
    -------

    Examples
    --------
    >>> dim = 1
    >>> level = 2
    >>> indices, grid_points, weights = sparse_gauss_hermite_quadrature(dim, level)
    >>> integration_result = np.sum(weights)
    >>> analytical_solution = np.power(np.pi, dim / 2)
    >>> np.isclose(integration_result, analytical_solution)
    True
    >>> dim = 2
    >>> indices, grid_points, weights = sparse_gauss_hermite_quadrature(dim, level)
    >>> integration_result = np.sum(weights)
    >>> analytical_solution = np.power(np.pi, dim / 2)
    >>> np.isclose(integration_result, analytical_solution)
    True

    """
    one_d_gh_nodes_and_weights_lists = one_d_gauss_hermite_nodes_up_to_a_level(level)
    smol_indices, grid_points, weights = smolyak_grid(dim, one_d_gh_nodes_and_weights_lists,
                                                      weight_cut_off=weight_cut_off)

    return smol_indices, grid_points, weights
