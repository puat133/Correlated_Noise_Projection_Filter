from collections.abc import Callable

import jax.numpy as jnp
import sympy as sp

from projection_filter.exponential_family_projection_filter_correlated_restricted import RestrictedCorrelatedProjectionFilter
from projection_filter.one_d_exponential_family import OneDExponentialFamily
from projection_filter.one_d_exponential_family_projection_filter import OneDExponentialFamilyProjectionFilter
from symbolic.n_d import get_projection_filter_matrices_correlated, remove_monoms_from_remaining_stats, \
    construct_remaining_statistics
from symbolic.one_d import SDE
from symbolic.sympy_to_jax import sympy_matrix_to_jax


class OneDimensionalRestrictedCorrelatedProjectionFilter(RestrictedCorrelatedProjectionFilter,
                                                         OneDExponentialFamilyProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 noise_correlation_matrix: jnp.ndarray,
                 nodes_number: int = 10,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray] = lambda x, params: jnp.arctanh(x),
                 ode_solver: str = 'euler',
                 bijection_parameters: tuple = None,
                 theta_indices_for_bijection_params=(jnp.array([0], dtype=jnp.int32), jnp.array([1], dtype=jnp.int32)),
                 moment_matching_iterations: int = 1
                 ):
        """

        Parameters
        ----------
        dynamic_sde
        measurement_sde
        natural_statistics_symbolic
        constants
        initial_condition
        measurement_record
        delta_t
        noise_correlation_matrix
        nodes_number
        bijection
        ode_solver
        bijection_parameters
        theta_indices_for_bijection_params
        moment_matching_iterations
        """
        self._theta_indices_for_bijection_params = theta_indices_for_bijection_params
        self._moment_matching_iterations: int = moment_matching_iterations
        self._S = noise_correlation_matrix
        self._monom_list = None
        self._remaining_monoms_list = None

        super(RestrictedCorrelatedProjectionFilter, self).__init__(
                                                       dynamic_sde=dynamic_sde,
                                                       measurement_sde=measurement_sde,
                                                       natural_statistics_symbolic=natural_statistics_symbolic,
                                                       constants=constants,
                                                       initial_condition=initial_condition,
                                                       measurement_record=measurement_record,
                                                       delta_t=delta_t,
                                                       nodes_number=nodes_number,
                                                       bijection=bijection,
                                                       ode_solver=ode_solver,
                                                       bijection_parameters=bijection_parameters)


        #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
        self._I = jnp.eye(measurement_record.shape[1])  # identity matrix of size m \times
        self._F0 = self._projection_filter_matrices[0]  # n_theta x m^2 x ntilde_theta
        self._F1 = self._projection_filter_matrices[1]  # n_theta x m x ntilde_theta
        self._F2 = self._projection_filter_matrices[2]  # m^2 x ntilde_theta
        self._F3 = self._projection_filter_matrices[3]  # m n_theta x ntilde_theta
        self._F4 = self._projection_filter_matrices[4]  # m^3 x ntilde_theta
        self._AR = self._projection_filter_matrices[5]  # m^2 x ntilde_theta
        self._A = self._projection_filter_matrices[6]  # n_theta x ntilde_theta
        self._H1 = self._projection_filter_matrices[7]  # m x n_theta
        self._H2 = self._projection_filter_matrices[8]  # m^2 x n_theta

        self._higher_stats_indices_from_fisher, updated_remaining_monom_list = \
            remove_monoms_from_remaining_stats(self.natural_statistics_symbolic,
                                               self._remaining_monoms_list,
                                               self._dynamic_sde)

        self._remaining_monom_list = updated_remaining_monom_list
        # sometimes self._remaining_monom_list becomes [] after above process, hence
        if self._remaining_monom_list:
            self._remaining_statistics_symbolic = self._construct_remaining_statistics()
            self._extended_statistics_symbolic = sp.Matrix.vstack(self._natural_statistics_symbolic,
                                                                  self._remaining_statistics_symbolic)
            self._remaining_statistics, _ = \
                sympy_matrix_to_jax(
                    self._remaining_statistics_symbolic, self._dynamic_sde.variables)
        else:
            self._extended_statistics_symbolic = self._natural_statistics_symbolic
            self._remaining_statistics = None

        self._remaining_statistics, _ = \
            sympy_matrix_to_jax(self._remaining_statistics_symbolic, [self._dynamic_sde.variables[0], ])

        self._exponential_density = OneDExponentialFamily(nodes_number, bijection, self._natural_statistics,
                                                          self._remaining_statistics,
                                                          bijection_parameters=bijection_parameters)
        if initial_condition.shape[0] != self._exponential_density.params_num:
            raise Exception("Wrong initial condition shape!, expected {} "
                            "given {}".format(initial_condition.shape[0], self._exponential_density.params_num))

    def _get_projection_filter_matrices(self):
        filter_matrices, monom_list, remaining_monoms_list = get_projection_filter_matrices_correlated(
            self.natural_statistics_symbolic,
            self._dynamic_sde,
            self._measurement_sde,
            self._S
        )
        self._monom_list = monom_list
        self._remaining_monoms_list = remaining_monoms_list
        return filter_matrices

    def _construct_remaining_statistics(self):
        return construct_remaining_statistics(self._dynamic_sde, self._remaining_monom_list)
