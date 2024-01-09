from sparse_quadrature.patterson_nodes_and_weights import NODES, WEIGHTS
from sparse_quadrature.smolyak import smolyak_grid, OneDNodesAndWeights
from sparse_quadrature.smolyax import OneDNodesAndWeightsX
import jax.numpy as jnp

def one_d_patterson_nodes_up_to_a_level(a_level: int) -> list[OneDNodesAndWeights]:
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
    https://people.math.sc.edu/Burkardt/m_src/patterson_rule/patterson_rule.html
    """

    one_d_nodes_and_weights_list = []
    if a_level > 8:
        raise Exception("Can only do up to eight.!")

    for i in range(a_level + 1):
        nodes = NODES[i]
        weights = WEIGHTS[i]
        one_d_nodes_and_weights_list.append(OneDNodesAndWeights(i + 1, nodes, weights))

    return one_d_nodes_and_weights_list


def one_d_patterson_nodes_up_to_a_levelx(a_level: int) -> list[OneDNodesAndWeightsX]:
    """
    The jax version

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
    https://people.math.sc.edu/Burkardt/m_src/patterson_rule/patterson_rule.html
    """

    one_d_nodes_and_weights_list = []
    if a_level > 8:
        raise Exception("Can only do up to eight.!")

    for i in range(a_level + 1):
        nodes = NODES[i]
        weights = WEIGHTS[i]
        one_d_nodes_and_weights_list.append(OneDNodesAndWeightsX(i + 1, jnp.array(nodes), jnp.array(weights)))

    return one_d_nodes_and_weights_list


def sparse_patterson_quadrature(dim: int, level: int):
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
    one_d_patterson_nodes_and_weights_lists = one_d_patterson_nodes_up_to_a_level(level)
    smol_indices, grid_points, weights = smolyak_grid(dim, one_d_patterson_nodes_and_weights_lists)

    return smol_indices, grid_points, weights
