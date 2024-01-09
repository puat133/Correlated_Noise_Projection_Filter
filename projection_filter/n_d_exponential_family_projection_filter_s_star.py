from itertools import chain
from collections.abc import Callable
from functools import partial
import jax.numpy as jnp
import numpy as onp
import sympy as sp
from jax import jit
from jax.lax import scan
from projection_filter.exponential_family_projection_filter_s_star import SStarProjectionFilter
from projection_filter.n_d_exponential_family_projection_filter import MultiDimensionalProjectionFilter
from symbolic.n_d import backward_diffusion, column_polynomials_coefficients, get_monomial_degree_set
from symbolic.one_d import SDE


class MultiDimensionalSStarProjectionFilter(MultiDimensionalProjectionFilter, SStarProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 nodes_number: int = 10,
                 integrator: str = 'spg',
                 level: int = 5,
                 epsilon: float = 0,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray] = lambda x, params: jnp.arctanh(x),
                 ode_solver: str = 'euler',
                 sRule: str = "gauss-patterson",
                 bijection_parameters: tuple = None,
                 weight_cut_off: float = None,
                 theta_indices_for_bijection_params=jnp.array([0, 1], dtype=jnp.int32),
                 moment_matching_iterations: int = 1
                 ):
        """
        A Class that encapsulates the S Bullet class of exponential family projection filter. The assumption is that
        the natural statistics span the measurement sde drift functions. Here we also assume that
        the sdes correspond to the dynamic and the measurement are polynomial functions with the same variable.
        The measurement numerical value is assumed to be close to the limit so that it can be treated as a continuous
        measurements. The filtering dynamics is solved using Euler-Maruyama scheme.
        See [1]

        Parameters
        ----------
        dynamic_sde : SDE
            SDE for the dynamic.

        measurement_sde : SDE
            SDE for the measurement.

        natural_statistics_symbolic : MutableDenseMatrix
            Natural statistics symbolic expression, at the moment, it only supports polynomial functions.

        constants : dict
            Some constants to be passed to the matrix expression.

        weight_cut_off: float
            weight cut off for the smolyak sparse grid

        Returns
        -------
        out : MultiDimensionalSStarProjectionFilter

        References
        ----------
        [1]

        """
        self._theta_indices_for_bijection_params = theta_indices_for_bijection_params
        self._moment_matching_iterations: int = moment_matching_iterations

        super().__init__(dynamic_sde,
                         measurement_sde,
                         natural_statistics_symbolic,
                         constants,
                         initial_condition,
                         measurement_record,
                         delta_t,
                         nodes_number,
                         integrator,
                         level,
                         epsilon,
                         bijection,
                         ode_solver,
                         sRule,
                         bijection_parameters,
                         weight_cut_off)

        #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
        L_mat = self._projection_filter_matrices[0]
        A_mat = self._projection_filter_matrices[1]
        b_mat = self._projection_filter_matrices[2]
        lambda_mat = self._projection_filter_matrices[3]
        self._L_0 = L_mat[:, 1:]
        self._ell_0 = L_mat[:, 0]
        self._A_0 = A_mat[:, 1:]
        self._b_h = b_mat[0, 1:]
        self._a_0 = A_mat[:, 0]
        self._b_0 = b_mat[0, 0]
        self._lamda = lambda_mat.T

    def _get_projection_filter_matrices(self) -> tuple[list[onp.ndarray],
                                                       list[tuple],
                                                       list[tuple]]:
        """
        Get matrices related to a projection filter with an exponential family densities.

        Returns
        -------
        matrices: tuple
            list of matrices containing M_0, m_h, and lamda
        """
        Lc = backward_diffusion(
            self._natural_statistics_symbolic, self._dynamic_sde)

        h_T_h = self._measurement_sde.drifts.transpose() * self._measurement_sde.drifts
        #   squared measurement drift times natural statistics
        h_T_h_per_2_times_c = h_T_h[0] * \
                              self._natural_statistics_symbolic
        Lc_min_h_T_h_times_c_per_2 = Lc - h_T_h_per_2_times_c / 2

        l_monom_set = get_monomial_degree_set(Lc, self._dynamic_sde.variables)

        a_monom_set = get_monomial_degree_set(
            Lc_min_h_T_h_times_c_per_2, self._dynamic_sde.variables)

        b_monom_set = get_monomial_degree_set(
            h_T_h / 2, self._dynamic_sde.variables)

        c_monom_set = get_monomial_degree_set(
            self._measurement_sde.drifts, self._dynamic_sde.variables)

        natural_monom_set = get_monomial_degree_set(
            self._natural_statistics_symbolic, self._dynamic_sde.variables)

        monom_set = natural_monom_set.union(a_monom_set).union(
            b_monom_set).union(c_monom_set).union(l_monom_set)

        remaining_monoms_set = monom_set.difference(natural_monom_set)
        constant_monom = (0, 0)
        constant_monom_list = [constant_monom, ]
        if constant_monom in remaining_monoms_set:
            remaining_monoms_set.remove(constant_monom)

        natural_monom_list = list(natural_monom_set)
        natural_monom_list.sort()
        remaining_monoms_list = list(remaining_monoms_set)
        remaining_monoms_list.sort()
        monom_list = list(chain.from_iterable(
            [constant_monom_list, natural_monom_list, remaining_monoms_list]))

        monoms_list_symbol_L, L_matrix = column_polynomials_coefficients(
            Lc, self._dynamic_sde.variables, monom_list)

        monoms_list_symbol_A, A_matrix = column_polynomials_coefficients(Lc_min_h_T_h_times_c_per_2,
                                                                         self._dynamic_sde.variables,
                                                                         monom_list)

        monoms_list_symbol_b, b_matrix = column_polynomials_coefficients(h_T_h / 2, self._dynamic_sde.variables,
                                                                         monom_list)

        monoms_list_symbol_lam, lambda_matrix = column_polynomials_coefficients(self._measurement_sde.drifts,
                                                                                self._dynamic_sde.variables,
                                                                                natural_monom_list)

        return [L_matrix, A_matrix, b_matrix, lambda_matrix], monom_list, remaining_monoms_list

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
        dummy_array = jnp.ones(self._moment_matching_iterations)

        def _to_be_scanned(_carry, _input):
            bijection_params_ = _carry
            eta_tilde_, fisher_ = self._get_eta_tilde_and_fisher(theta, bijection_params_)
            eta_ = eta_tilde_[:self.exponential_density.params_num]
            bijection_params_ = self.update_bijection_params(eta_, fisher_, bijection_params_)
            return bijection_params_, bijection_params_

        bijection_params, bijection_params_history = scan(_to_be_scanned, bijection_params, dummy_array)
        eta_tilde, fisher = self._get_eta_tilde_and_fisher(theta, bijection_params)
        return bijection_params, eta_tilde, fisher


    @partial(jit, static_argnums=[0, ])
    def update_bijection_params_vanilla(self,
                                        eta: jnp.array,
                                        fisher: jnp.array,
                                        old_bijection_params: tuple):
        _mu = jnp.take(eta, self._theta_indices_for_bijection_params[0])
        # _Sigma = fisher[self._indices_grid]
        _Sigma = eta[self._theta_indices_for_bijection_params[1]] - jnp.outer(_mu, _mu)
        _Sigma_eigvals, _Sigma_eigvects = jnp.linalg.eigh(_Sigma)

        # do not update scale factor
        _, _, _, scale_factor = old_bijection_params

        new_bijection_params = _mu, _Sigma_eigvals, _Sigma_eigvects, scale_factor
        return new_bijection_params