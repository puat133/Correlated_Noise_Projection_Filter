from collections.abc import Callable

import jax.numpy as jnp
from LowDiscrepancy import halton
from jax.tree_util import register_pytree_node_class

from projection_filter.n_d_exponential_family import MultiDimensionalExponentialFamily


@register_pytree_node_class
class MultiDimensionalExponentialFamilyQMC(MultiDimensionalExponentialFamily):
    def __init__(self, sample_space_dimension: int,
                 nodes_number: int,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 bijection_parameters: tuple = None
                 ):
        """
        Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
        of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

        Parameters
        ----------
        sample_space_dimension
        nodes_number
        bijection
        statistics
        remaining_statistics
        """

        super().__init__(sample_space_dimension=sample_space_dimension,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters)

        # This is the halton point which happen in hypercube [0,1]^d
        halton_points = halton(jnp.arange(nodes_number + 1), self._sample_space_dim)

        # the qmc_points are in [-1,1]^d for convenience
        self._quadrature_points = 2 * halton_points[1:] - 1
        self._bijected_points = self._bijection(self._quadrature_points, self.bijection_params)
        self._bijected_volume = jnp.reshape(self._log_dvolume(self._quadrature_points, self.bijection_params),
                                            (self._quadrature_points.shape[0], 1))

        self._nodes_num = self._quadrature_points.shape[0]

    @property
    def srule(self):
        return "qmc"

    @property
    def bijected_points(self):
        return self._bijected_points

    @property
    def quadrature_points(self):
        return self._quadrature_points

    @property
    def nodes_number(self):
        return self._nodes_num

    def numerical_integration(self, numerical_values: jnp.ndarray, axis=-1):
        # since the domain is changed to $(-1,1)^d$ the integration becomes like this
        return jnp.mean(numerical_values, axis=axis) * (jnp.power(2, self._sample_space_dim))

    def _compute_D_part_for_fisher_metric(self, c: jnp.ndarray, exp_c_theta: jnp.ndarray, dvol: jnp.ndarray):
        D = jnp.einsum('k,ki,kj,k', exp_c_theta.ravel(), c, c, dvol) / \
            self._quadrature_points.shape[
                0]
        return D

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
