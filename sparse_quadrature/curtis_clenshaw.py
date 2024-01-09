import numpy as np
import numba as nb
from .smolyak import OneDNodesAndWeights, smolyak_grid


@nb.njit(fastmath=True)
def clenshaw_curtis_nodes_and_weight(n: int):
    """
    Generate Curtis--Clenshaw one dimensional quadrature nodes and weight

    Parameters
    ----------
    n   : int
        number of nodes

    Returns
    -------
    x: np.ndarray
        quadrature nodes
    w: np.ndarray
        weights

    Examples
    --------

    """
    x = np.zeros(n)
    w = np.zeros(n)

    if n == 1:
        w[0] = 2.0
    else:
        theta = np.zeros(n)
        for i in range(n):
            theta[i] = float(n - 1 - i) * np.pi / float(n - 1)
        x = np.cos(theta)
        w = np.zeros(n)
        for i in range(n):
            w[i] = 1.0
            jhi = ((n - 1) // 2)
            for j in range(0, jhi):
                if 2 * (j + 1) == (n - 1):
                    b = 1.0
                else:
                    b = 2.0

            w[i] = w[i] - b * np.cos(2.0 * (j + 1) * theta[i]) / (4 * j * (j + 2) + 3)

        w[0] = w[0] / (n - 1)
        for i in range(1, n - 1):
            w[i] = 2.0 * w[i] / (n - 1)
        w[n - 1] = w[n - 1] / (n - 1)
    return x, w


@nb.njit(fastmath=True)
def one_d_clenshaw_curtis_nodes_up_to_a_level(a_level: int) -> list[OneDNodesAndWeights]:
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
    https://people.math.sc.edu/Burkardt/m_src/sparse_grid_cc/sparse_grid_cc.html
    """

    one_d_nodes_and_weights_list = []
    polynomial_orders = np.power(2, np.arange(a_level + 1)) + 1
    polynomial_orders[0] = 1

    for i in range(a_level + 1):
        polynomial_order = polynomial_orders[i]
        nodes, weights = clenshaw_curtis_nodes_and_weight(polynomial_order)
        one_d_nodes_and_weights_list.append(OneDNodesAndWeights(i + 1, nodes, weights))

    return one_d_nodes_and_weights_list


def sparse_clenshaw_curtis_quadrature(dim: int, level: int):
    """

    Parameters
    ----------
    dim
    level

    Returns
    -------

    Examples
    --------
    """
    one_d_gh_nodes_and_weights_lists = one_d_clenshaw_curtis_nodes_up_to_a_level(level)
    smol_indices, grid_points, weights = smolyak_grid(dim, one_d_gh_nodes_and_weights_lists)

    return smol_indices, grid_points, weights
