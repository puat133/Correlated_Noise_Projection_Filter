from collections.abc import Callable
from jax.tree_util import register_pytree_node_class
from projection_filter.n_d_exponential_family import MultiDimensionalExponentialFamily
import jax.numpy as jnp
from abc import ABC, abstractmethod


@register_pytree_node_class
class MultiDimensionalExponentialFamilySparse(MultiDimensionalExponentialFamily, ABC):
    def __init__(self, sample_space_dimension: int,
                 sparse_grid_level: int,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 bijection_parameters: tuple = None,
                 epsilon: float = 0,
                 sRule: str = "",
                 weight_cut_off: float = None
                 ):
        """
        Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
        of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

        Parameters
        ----------
        sample_space_dimension : int
            sample space dimension
        sparse_grid_level : int
            sparse grid level
        statistics : Callable[[jnp.ndarray], jnp.ndarray]
            statistics function
        remaining_statistics: Callable[[jnp.ndarray], jnp.ndarray]
            remaining statistics function
        weight_cut_off: float
            weight cut off for the smolyak sparse grid
        bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray]
            bijection function
        bijection_parameters: tuple
            bijection parameters
        epsilon : float
            epsilon
        sRule   : str
            sparse integration rule
        weight_cut_off: float
            weight cut off for the smolyak sparse grid
        """

        super().__init__(sample_space_dimension=sample_space_dimension,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters)
        self._spg_level = sparse_grid_level
        self._srule = sRule
        self._epsilon = epsilon
        self._weight_cut_off = weight_cut_off
        points, weights = self.initialize_sparse_grid()

        # To avoid nan in bijection result
        # Gauss-Hermite Quadrature is not confined in [-1.1]
        if self._srule.lower() != "gauss-hermite":
            mask = jnp.linalg.norm(points, ord=jnp.inf, axis=-1) < 1 - self._epsilon
            self._spg_weights = jnp.asarray(weights[mask])
            self._quadrature_points = jnp.asarray(points[mask])
        else:
            self._spg_weights = jnp.asarray(weights)
            self._quadrature_points = jnp.asarray(points)

        self._nodes_num = self._spg_weights.shape[0]
        self._bijected_points = self._bijection(self._quadrature_points, self.bijection_params)
        self._bijected_log_dvolume = jnp.reshape(self._log_dvolume(self._quadrature_points, self.bijection_params),
                                                 (self._quadrature_points.shape[0], 1))

    @abstractmethod
    def initialize_sparse_grid(self):
        return [], []

    @property
    def srule(self):
        return self._srule

    @property
    def spg_weights(self):
        return self._spg_weights

    @property
    def nodes_number(self):
        return self._nodes_num

    @property
    def bijected_points(self):
        return self._bijected_points

    @property
    def quadrature_points(self):
        return self._quadrature_points

    @property
    def sparse_grid_level(self):
        return self._spg_level

    def numerical_integration(self, numerical_values: jnp.ndarray, axis=None):
        if not axis:
            result = numerical_values  @ self._spg_weights
        else:
            result = jnp.mean(numerical_values  @ self._spg_weights, axis=axis)
        return result

    def _compute_D_part_for_fisher_metric(self, c: jnp.ndarray, exp_c_theta: jnp.ndarray, dvol: jnp.ndarray):
        D = jnp.einsum('k,ki,kj,k', exp_c_theta.ravel(), c, c, dvol.ravel() * self._spg_weights)
        return D

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
