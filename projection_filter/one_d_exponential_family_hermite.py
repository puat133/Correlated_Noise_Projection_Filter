from collections.abc import Callable

import jax.numpy as jnp
from jax import jit
from scipy.special import roots_hermite

from projection_filter.one_d_exponential_family import OneDExponentialFamily


@jit
def gauss_hermite_bijection(xi: jnp.ndarray, bijection_params: tuple):
    _mu, _Var, _scale_factor = bijection_params
    return _mu + _scale_factor * jnp.sqrt(2 * _Var) * xi


class HermiteOneDExponentialFamily(OneDExponentialFamily):
    def __init__(self, nodes_number: int,
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 bijection_parameters: tuple = None):
        super().__init__(nodes_number,
                         gauss_hermite_bijection,
                         statistics,
                         remaining_statistics,
                         bijection_parameters)

        # Here instead of using Gauss-Chebyshev quadrature, we use Gauss-Hermite quadrature
        hermite_root_, hermite_weight_ = roots_hermite(self.nodes_number)
        self._weight = jnp.array(hermite_weight_)
        # Hermit roots are not on (-1,1) but they are on R
        self._abscissa = jnp.array(hermite_root_)

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
        x_tilde = self._abscissa

        bijected_x_tilde = self._bijection(x_tilde, bijection_params)
        # This is the Gauss-Hermite
        inner = self._log_dvolume(x_tilde, bijection_params) + \
                self._natural_statistics(bijected_x_tilde) @ theta + x_tilde ** 2

        max_inner = jnp.max(inner)
        normalized_par_int =  jnp.exp(inner - max_inner)
        res = jnp.sum(normalized_par_int * self._weight)

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
        x_tilde = self._abscissa
        res = jnp.sum(fun(self._bijection(x_tilde, bijection_params)) *
                      jnp.exp(self._log_dvolume(x_tilde, bijection_params)) *
                      jnp.exp(x_tilde ** 2) * self._weight, axis=-1)
        return res

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
        x_tilde = self._abscissa
        bijected_x_tilde = self._bijection(x_tilde, bijection_params)
        # This is the Gauss-Hermite
        inner = self._log_dvolume(x_tilde, bijection_params) + \
                self._extended_statistics(bijected_x_tilde) @ theta_extended + x_tilde ** 2

        max_inner = jnp.max(inner)
        normalized_par_int =  jnp.exp(inner - max_inner)
        res = jnp.sum(normalized_par_int *
                      self._weight)

        return res, max_inner

    @property
    def weight(self):
        return self._weight
