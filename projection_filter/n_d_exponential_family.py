from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class

from projection_filter.exponential_family import ExponentialFamily


@register_pytree_node_class
class MultiDimensionalExponentialFamily(ExponentialFamily, ABC):
    def __init__(self, sample_space_dimension: int,
                 bijection: Callable[[jnp.ndarray], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 statistics_vectorization_signature: str = '(d)->(n)',
                 partition_vectorization_signature: str = '(m)->()',
                 bijection_parameters: tuple = None
                 ):
        """
        Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
        of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

        Parameters
        ----------
        sample_space_dimension
        bijection
        statistics
        remaining_statistics
        statistics_vectorization_signature
        partition_vectorization_signature
        """

        super().__init__(sample_space_dimension=sample_space_dimension,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         statistics_vectorization_signature=statistics_vectorization_signature,
                         partition_vectorization_signature=partition_vectorization_signature,
                         bijection_parameters=bijection_parameters)

        # default value to be realized in the implemented class
        self._quadrature_points = jnp.empty((1,))
        self._bijected_points = jnp.empty((1,))
        self._bijected_log_dvolume = jnp.empty((1,))
        self._srule = ""

    @property
    @abstractmethod
    def srule(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def bijected_points(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def quadrature_points(self):
        raise NotImplementedError

    @abstractmethod
    def numerical_integration(self, numerical_values: jnp.ndarray, axis=None):
        raise NotImplementedError

    @partial(jit, static_argnums=[0, ])
    def integrate_partition(self, theta: jnp.ndarray, bijection_params: tuple) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_tilde = self._quadrature_points
        inner = self._log_dvolume(x_tilde, bijection_params) + \
                self._natural_statistics(self._bijection(x_tilde, bijection_params)) @ theta
        max_inner = jnp.max(inner)
        normalized_par_int = jnp.exp(inner - max_inner)

        res = self.numerical_integration(normalized_par_int)
        return res, max_inner

    @partial(jit, static_argnums=[0, ])
    def integrate_partition_extended(self, theta_extended: jnp.ndarray, bijection_params: tuple) -> \
            tuple[jnp.ndarray, jnp.ndarray]:
        x_tilde = self._quadrature_points
        inner = self._log_dvolume(x_tilde, bijection_params) + \
                self._extended_statistics(self._bijection(x_tilde, bijection_params)) @ theta_extended
        max_inner = jnp.max(inner)
        normalized_par_int = jnp.exp(inner - max_inner)
        res = self.numerical_integration(normalized_par_int)
        return res, max_inner

    def get_density_values(self, grid_limits: jnp.ndarray, theta: jnp.ndarray, nb_of_points: jnp.ndarray
                           , bijection_params: tuple) -> \
            tuple[jnp.ndarray, jnp.ndarray]:
        x_ = []
        for i in range(self._sample_space_dim):
            temp_ = jnp.linspace(grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
            x_.append(temp_)
        grids = jnp.meshgrid(*x_, indexing='xy')
        grids = jnp.stack(grids, axis=-1)

        return self.get_density_values_from_grids(grids, theta, bijection_params)

    def get_density_values_from_grids(self, grids: jnp.ndarray, theta: jnp.ndarray, bijection_params: tuple):
        c_ = self.natural_statistics(grids)

        @jit
        def _evalulate_density(theta_):
            psi_ = self.log_partition(theta_, bijection_params)
            density_ = jnp.exp(c_ @ theta_ - psi_)
            return density_

        density = _evalulate_density(theta)
        return grids, density

    @abstractmethod
    def _compute_D_part_for_fisher_metric(self, c: jnp.ndarray, exp_c_theta: jnp.ndarray, dvol: jnp.ndarray):
        raise NotImplementedError

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
        integrand = jnp.exp(self._log_dvolume(x_tilde, bijection_params)) * fun(
            self.bijection(x_tilde, bijection_params))
        return self.numerical_integration(integrand)

    def direct_fisher_metric(self, theta: jnp.ndarray, bijection_param):
        """
        Compute Fisher metric without using autodiff. This will produces exactly the same result with
        self.fisher_metric.

        Parameters
        ----------
        theta
        bijection_param

        Returns
        -------

        """
        x_tilde = self._quadrature_points
        c = self._natural_statistics(self._bijection(x_tilde, bijection_param))
        inner = c @ theta
        max_inner = jnp.max(inner)  # exponential max_inner will cancels out
        exp_c_theta = jnp.reshape(jnp.exp(inner - max_inner), (self._quadrature_points.shape[0], 1))
        dvol = self._log_dvolume(x_tilde, bijection_param)[:, jnp.newaxis]
        expectation_of_exp_c_theta_dv = self.numerical_integration((exp_c_theta * dvol).T)
        expectation_of_exp_c_theta_c_dv = self.numerical_integration((exp_c_theta * c * dvol).T, axis=0)
        D = self._compute_D_part_for_fisher_metric(c, exp_c_theta, dvol)
        return (1 / expectation_of_exp_c_theta_dv) * ((-1 / expectation_of_exp_c_theta_dv) *
                                                      jnp.outer(expectation_of_exp_c_theta_c_dv,
                                                                expectation_of_exp_c_theta_c_dv) + D)

    def direct_natural_statistics_expectation(self, theta: jnp.ndarray, bijection_param):
        """
        Compute natural statistics expectation without using autodiff. This will produces exactly the same result with
        self.natural_statistics_expectation.
        Parameters
        ----------
        theta
        bijection_param
        Returns
        -------

        """
        x_tilde = self._quadrature_points
        c = self._natural_statistics(self._bijection(x_tilde, bijection_param))
        inner = c @ theta
        max_inner = jnp.max(inner)  # exponential max_inner will cancels out
        exp_c_theta = jnp.reshape(jnp.exp(inner - max_inner), (self._quadrature_points.shape[0], 1))
        dvol = self._log_dvolume(x_tilde, bijection_param)[:, jnp.newaxis]
        expectation_of_exp_c_theta_dv = self.numerical_integration((exp_c_theta * dvol).T)
        expectation_of_exp_c_theta_c_dv = self.numerical_integration((exp_c_theta * c * dvol).T, axis=0)
        return expectation_of_exp_c_theta_c_dv / expectation_of_exp_c_theta_dv

    def direct_extended_statistics_expectation(self, theta_extended: jnp.ndarray, bijection_param):
        """
        Compute extended statistics expectation without using autodiff. This will produces exactly the same result with
        self.extended_statistics_expectation.
        Parameters
        ----------
        theta_extended
        bijection_param

        Returns
        -------

        """
        x_tilde = self._quadrature_points
        c = self._extended_statistics(self._bijection(x_tilde, bijection_param))
        inner = c @ theta_extended
        max_inner = jnp.max(inner)  # exponential max_inner will cancels out
        exp_c_theta = jnp.reshape(jnp.exp(inner - max_inner), (self._quadrature_points.shape[0], 1))
        dvol = self._log_dvolume(x_tilde, bijection_param)[:, jnp.newaxis]
        expectation_of_exp_c_theta_dv = self.numerical_integration((exp_c_theta * dvol).T)
        expectation_of_exp_c_theta_c_dv = self.numerical_integration((exp_c_theta * c * dvol).T, axis=0)
        return expectation_of_exp_c_theta_c_dv / expectation_of_exp_c_theta_dv

    def tree_flatten(self):
        auxiliaries = None
        return (self._sample_space_dim, self._nodes_num, self._params_num, self.bijection_params), auxiliaries

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
