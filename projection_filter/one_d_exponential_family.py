from collections.abc import Callable

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from projection_filter.exponential_family import ExponentialFamily


@register_pytree_node_class
class OneDExponentialFamily(ExponentialFamily):
    def __init__(self, nodes_number: int,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 bijection_parameters: tuple = None):
        self._nodes_num = nodes_number

        # This is the Chebyshev node positions
        self._abscissa = jnp.cos(
            jnp.pi * (jnp.arange(self._nodes_num) + 0.5) / self._nodes_num)
        self._abscissa = self._abscissa[:, jnp.newaxis]
        self._inverse_weight = jnp.sqrt(1 - self._abscissa * self._abscissa).squeeze()

        super().__init__(sample_space_dimension=1,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters
                         )

    @property
    def nodes_number(self):
        return self._nodes_num

    @property
    def abscissa(self):
        return self._abscissa

    def get_density_values(self, grid_limits: jnp.ndarray, theta: jnp.ndarray, nb_of_points: jnp.ndarray,
                           bijection_params: tuple) -> \
            tuple[jnp.ndarray, jnp.ndarray]:
        grid_limits = grid_limits.squeeze()
        grid = jnp.linspace(
            grid_limits[0], grid_limits[1], nb_of_points[0], endpoint=True)
        density = self.evaluate_density_at_points(grid[:,jnp.newaxis], theta, bijection_params)
        return grid, density

    def evaluate_density_at_points(self, x: jnp.ndarray, theta: jnp.ndarray, bijection_params: tuple):
        c_ = self._natural_statistics(x)
        psi_ = self.log_partition(theta, bijection_params)
        density = jnp.exp(c_ @ theta - psi_)
        return density

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
        x_tilde = self._abscissa
        res = jnp.sum(fun(self._bijection(x_tilde, bijection_params)).squeeze() *
                      jnp.exp(self._log_dvolume(x_tilde, bijection_params)) * self._inverse_weight, axis=-1) \
              * jnp.pi / self._nodes_num
        return res

    def integrate_partition(self, theta: jnp.ndarray, bijection_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform integration of the log partition function using Chebyshev rule

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
        x_tilde = self._abscissa

        # This is the Chebyshev
        inner = self._log_dvolume(x_tilde, bijection_params) + self._natural_statistics(
            self._bijection(x_tilde, bijection_params)) @ theta
        max_inner = jnp.max(inner)
        normalized_par_int = jnp.exp(inner - max_inner)
        res = jnp.sum(normalized_par_int *
                      self._inverse_weight) * jnp.pi / self._nodes_num

        return res, max_inner

    def integrate_partition_extended(self, theta_extended: jnp.ndarray, bijection_params: jnp.ndarray) -> \
            tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform integration of the log partition extended function using Chebyshev rule

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
        x_tilde = self._abscissa

        # This is the Chebyshev
        inner = self._log_dvolume(x_tilde, bijection_params) + \
                self._extended_statistics(self._bijection(x_tilde, bijection_params)) @ theta_extended
        max_inner = jnp.max(inner)
        normalized_par_int = jnp.exp(inner - max_inner)
        res = jnp.sum(normalized_par_int *
                      self._inverse_weight) * jnp.pi / self._nodes_num

        return res, max_inner

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
