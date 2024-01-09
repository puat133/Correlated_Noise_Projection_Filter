from collections.abc import Callable
from jax.tree_util import register_pytree_node_class
from projection_filter.n_d_exponential_family_sparse import MultiDimensionalExponentialFamilySparse
from sparse_quadrature import sparse_gauss_hermite_quadrature
import jax.numpy as jnp
from jax import jit
from utils.vectorized import inner


def plain_gauss_hermite_bijection(xtilde: jnp.ndarray, bijection_params: tuple) -> jnp.ndarray:
    # since _Sigma_eigvects transpose is equal to its inverse
    _mu, _Sigma_eigvals, _Sigma_eigvects, scale_factor = bijection_params
    return _mu + scale_factor * jnp.sqrt(2) * _Sigma_eigvects.T@(jnp.sqrt(_Sigma_eigvals) * xtilde)


gauss_hermite_bijection = jit(jnp.vectorize(plain_gauss_hermite_bijection, signature='(n)->(n)', excluded=(1,)))


@register_pytree_node_class
class MultiDimensionalExponentialFamilyHermite(MultiDimensionalExponentialFamilySparse):
    def __init__(self, sample_space_dimension: int,
                 sparse_grid_level: int,
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 bijection_parameters: tuple = None,
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
        bijection_parameters: tuple
            bijection parameters
        weight_cut_off: float
            weight cut off for the smolyak sparse grid
        """

        super().__init__(sample_space_dimension=sample_space_dimension,
                         sparse_grid_level=sparse_grid_level,
                         bijection=gauss_hermite_bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters,
                         epsilon=0,
                         sRule="gauss-hermite",
                         weight_cut_off=weight_cut_off)

    def initialize_sparse_grid(self):
        _, points, weights = sparse_gauss_hermite_quadrature(self._sample_space_dim,
                                                             self._spg_level,
                                                             weight_cut_off=self._weight_cut_off)
        return points, weights

    def integrate_partition(self, theta: jnp.ndarray, bijection_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform integration of the log partition function using Gauss-Hermite rule

        Parameters
        ----------
        theta: ndarray
            Parameter of the extended exponential family

        bijection_params: ndarray
            bijection parameters

        Returns
        -------
        res: ndarray
            Integration result

        """

        bijected_x_points = self._bijection(self._quadrature_points, bijection_params)

        # This is the Gauss-Hermite
        inner_ = self._log_dvolume(self._quadrature_points, bijection_params) + \
                 self._natural_statistics(bijected_x_points) @ theta + inner(self._quadrature_points,
                                                                             self._quadrature_points)

        max_inner = jnp.max(inner_)
        normalized_par_int = jnp.exp(inner_ - max_inner)
        res = self.numerical_integration(normalized_par_int)

        return res, max_inner

    def integrate_partition_extended(self, theta_extended: jnp.ndarray, bijection_params: jnp.ndarray) -> \
            tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform integration of the log partition extended function using Gauss-Hermite rule

        Parameters
        ----------
        theta_extended: ndarray
            Parameter of the extended exponential family

        bijection_params: ndarray
            bijection parameters

        Returns
        -------
        res: ndarray
            Integration result

        """

        bijected_x_points = self._bijection(self._quadrature_points, bijection_params)
        # This is the Gauss-Hermite
        inner_ = self._log_dvolume(self._quadrature_points, bijection_params) + \
                 self._extended_statistics(bijected_x_points) @ theta_extended + \
                 inner(self._quadrature_points, self._quadrature_points)

        max_inner = jnp.max(inner_)
        normalized_par_int = jnp.exp(inner_ - max_inner)
        res = self.numerical_integration(normalized_par_int)

        return res, max_inner

    def integrate(self, fun: Callable[[jnp.ndarray, ], jnp.ndarray], bijection_params: tuple) -> jnp.ndarray:
        """
        Perform integration of a given function

        Parameters
        ----------
        fun
        bijection_params

        Returns
        -------

        """
        x_tilde = self._quadrature_points
        integrand = jnp.exp(self._log_dvolume(x_tilde, bijection_params)
                            + inner(x_tilde, x_tilde)) * fun(self.bijection(x_tilde, bijection_params))
        return self.numerical_integration(integrand)
