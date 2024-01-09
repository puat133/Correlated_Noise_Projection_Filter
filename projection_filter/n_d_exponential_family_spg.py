from collections.abc import Callable

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from sparse_quadrature.curtis_clenshaw import sparse_clenshaw_curtis_quadrature
from sparse_quadrature.kronrod import sparse_kronrod_quadrature
from sparse_quadrature.patterson import sparse_patterson_quadrature
from sparse_quadrature.gauss_legendre import sparse_gauss_legendre_quadrature
from projection_filter.n_d_exponential_family_sparse import MultiDimensionalExponentialFamilySparse


@register_pytree_node_class
class MultiDimensionalExponentialFamilySPG(MultiDimensionalExponentialFamilySparse):
    def __init__(self, sample_space_dimension: int,
                 sparse_grid_level: int,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 epsilon: float = 1e-7,
                 sRule: str = "gauss-patterson",
                 bijection_parameters: tuple = None
                 ):
        """
        Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
        of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

        Parameters
        ----------
        sample_space_dimension
        sparse_grid_level
        bijection
        statistics
        remaining_statistics
        epsilon
        sRule
        bijection_parameters
        """
        self._epsilon = epsilon

        if sRule.lower() in ["clenshaw-curtis", "gauss-kronrod", "gauss-legendre", "gauss-patterson"]:
            if sRule == "gauss-patterson" and sparse_grid_level > 8:
                sparse_grid_level = 8
            sRule = sRule
        else:
            sRule = "clenshaw-curtis"

        super().__init__(sample_space_dimension=sample_space_dimension,
                         sparse_grid_level=sparse_grid_level,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters,
                         sRule=sRule)

    def initialize_sparse_grid(self):
        if self._srule.lower() == "gauss-kronrod":
            _, points, weights = sparse_kronrod_quadrature(self._sample_space_dim,
                                                           self._spg_level)
        elif self._srule.lower() == "gauss-patterson":
            _, points, weights = sparse_patterson_quadrature(self._sample_space_dim,
                                                             self._spg_level)
        elif self._srule.lower() == "gauss-legendre":
            _, points, weights = sparse_gauss_legendre_quadrature(self._sample_space_dim,
                                                             self._spg_level)
        else:
            _, points, weights = sparse_clenshaw_curtis_quadrature(self._sample_space_dim,
                                                                   self._spg_level)
        return points, weights
