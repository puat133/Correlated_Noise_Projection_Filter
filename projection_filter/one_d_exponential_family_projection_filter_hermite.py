from projection_filter.one_d_exponential_family_hermite import HermiteOneDExponentialFamily
from symbolic.one_d import SDE
from functools import partial
from jax import jit
import jax.numpy as jnp
import sympy as sp
from jax.lax import scan
from projection_filter.one_d_exponential_family_projection_filter_s_star import OneDimensionalSStarProjectionFilter
from projection_filter.one_d_exponential_family_hermite import gauss_hermite_bijection


class HermiteOneDimensionalSStarProjectionFilter(OneDimensionalSStarProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 bijection_parameters: tuple,
                 nodes_number: int = 8,
                 ode_solver: str = 'euler',
                 theta_indices_for_bijection_params=(jnp.array([0], dtype=jnp.int32), jnp.array([1], dtype=jnp.int32)),
                 moment_matching_iterations: int = 1):
        super().__init__(dynamic_sde=dynamic_sde,
                         measurement_sde=measurement_sde,
                         natural_statistics_symbolic=natural_statistics_symbolic,
                         constants=constants,
                         initial_condition=initial_condition,
                         measurement_record=measurement_record,
                         delta_t=delta_t,
                         nodes_number=nodes_number,
                         bijection=gauss_hermite_bijection,
                         ode_solver=ode_solver,
                         bijection_parameters=bijection_parameters,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params,
                         moment_matching_iterations=moment_matching_iterations)

        self._exponential_density = HermiteOneDExponentialFamily(nodes_number, self._natural_statistics,
                                                                 self._remaining_statistics,
                                                                 bijection_parameters=bijection_parameters)

    @partial(jit, static_argnums=[0, ])
    def compute_eta_tilde_fisher_and_update_bijection_parameters(self, theta: jnp.ndarray,
                                                                 bijection_params: tuple) -> tuple:
        """
        This to be used to update the bijection parameters based on given theta. At this generic class it does nothing.
        On the implementation, this function can be modified.

        Parameters
        ----------
        theta   : ndarray
            Exponential density natural parameters.
        bijection_params : ndarray
            Exponential family bijection parameters.

        Returns
        -------
        new_bijection_params : ndarray
        """
        # update bijection parameters using moment matching n times
        dummy_array = jnp.ones(self._moment_matching_iterations)

        def _to_be_scanned(_carry,_input):
            bijection_params_ = _carry
            eta_tilde_, fisher_ = self._get_eta_tilde_and_fisher(theta, bijection_params_)
            eta_ = eta_tilde_[:self.exponential_density.params_num]
            bijection_params_ = self.update_bijection_params(eta_, fisher_, bijection_params_)
            return bijection_params_, None

        bijection_params, _ = scan(_to_be_scanned, bijection_params, dummy_array)
        eta_tilde, fisher = self._get_eta_tilde_and_fisher(theta, bijection_params)
        return bijection_params, eta_tilde, fisher

    @partial(jit, static_argnums=[0, ])
    def update_bijection_params(self, eta: jnp.array, fisher: jnp.array, old_bijection_params: tuple):
        _mu = jnp.take(eta, self._theta_indices_for_bijection_params[0])
        _Var = jnp.take(eta, self._theta_indices_for_bijection_params[1]) - _mu ** 2
        new_bijection_params = _mu[0], _Var[0], old_bijection_params[-1]
        return new_bijection_params
