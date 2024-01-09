from abc import ABC, abstractmethod
from functools import partial
from collections.abc import Callable

import jax.numpy as jnp
import sympy as sp
from jax import jit
from jax.lax import scan
from projection_filter.one_d_exponential_family import OneDExponentialFamily
from projection_filter.exponential_family_projection_filter import ExponentialFamilyProjectionFilter
from symbolic.one_d import SDE
from symbolic.sympy_to_jax import sympy_matrix_to_jax


class OneDExponentialFamilyProjectionFilter(ExponentialFamilyProjectionFilter, ABC):
    def __init__(self,
                 dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 nodes_number: int = 10,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray] = lambda x, params: jnp.arctanh(x),
                 ode_solver: str = 'euler',
                 bijection_parameters: tuple = None,
                 rescale_measurement: bool = True
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

        Returns
        -------
        out : OneDimensionalSStarProjectionFilter

        References
        ----------
        [1]

        """

        # this need to be hardcoded to ensure everything goes ok.
        ExponentialFamilyProjectionFilter.__init__(self,
                                                   dynamic_sde,
                                                   measurement_sde,
                                                   natural_statistics_symbolic,
                                                   constants,
                                                   initial_condition,
                                                   measurement_record,
                                                   delta_t,
                                                   bijection,
                                                   ode_solver=ode_solver,
                                                   bijection_parameters=bijection_parameters,
                                                   rescale_measurement=rescale_measurement)

        #       the statistics are assumed to be free of additional symbolic parameters
        self._natural_statistics, _ = \
            sympy_matrix_to_jax(self._natural_statistics_symbolic, [self._dynamic_sde.variables[0], ], squeeze=True)

        projection_filter_matrices_ = self._get_projection_filter_matrices()

        if projection_filter_matrices_:  # projection_filter_matrices_ can be None for the case
            # OneDimensionalCorrelatedProjectionFilter
            #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
            self._projection_filter_matrices = projection_filter_matrices_
        else:
            self._extended_statistics_symbolic = self._natural_statistics_symbolic
            self._remaining_statistics = None

        self._current_state = initial_condition
        self._state_history = self._current_state[jnp.newaxis, :]
        self._nodes_num = nodes_number
        self._exponential_density = OneDExponentialFamily(self._nodes_num,
                                                          bijection,
                                                          self._natural_statistics,
                                                          self._remaining_statistics,
                                                          bijection_parameters=bijection_parameters)

    @property
    def natural_statistics(self):
        return self._exponential_density.natural_statistics

    @property
    def remaining_statistics(self):
        return self._exponential_density.remaining_statistics

    @property
    def exponential_density(self):
        return self._exponential_density

    def get_density_values(self, grid_limits, nb_of_points):
        grid_limits = grid_limits.squeeze()
        x_ = jnp.linspace(grid_limits[0], grid_limits[1], nb_of_points[0], endpoint=True)
        c_ = self._exponential_density.natural_statistics(x_[:, jnp.newaxis])
        abscissa = self._exponential_density.abscissa

        @jit
        def _evaluate_density_loop(carry_, input_):
            theta_, bijection_params_ = input_
            psi_ = self._exponential_density.log_partition(theta_, bijection_params_)
            density = jnp.exp(c_ @ theta_ - psi_)
            bijected_abscissa = self._exponential_density.bijection(abscissa, bijection_params_)
            return None, (density, bijected_abscissa)

        _, (density_history_, bijected_abscissa_history_) = scan(_evaluate_density_loop, None,
                                                                 (
                                                                     self.state_history,
                                                                     self.bijection_parameters_history))
        return x_, density_history_, bijected_abscissa_history_

    @abstractmethod
    def _construct_remaining_statistics(self):
        NotImplementedError

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
        eta_tilde, fisher = self._get_eta_tilde_and_fisher(theta, bijection_params)
        return bijection_params, eta_tilde, fisher

    @partial(jit, static_argnums=[0, ])
    def get_eta_tilde_and_eta_from_eta_extended_and_fisher(self, eta_extended_, fisher_):
        eta_ = eta_extended_[:self._exponential_density.params_num]
        eta_high_ = eta_extended_[self._exponential_density.params_num:]
        # expected value of additional natural statistics of the form c_i*c_j
        # eta_combination_ = fisher_[self._higher_stats_indices_from_fisher] + \
        #                    eta_[self._higher_stats_indices_from_fisher[0]] * eta_[
        #                        self._higher_stats_indices_from_fisher[1]]
        # eta_tilde_ = jnp.concatenate((eta_, eta_combination_, eta_high_))
        eta_tilde_ = jnp.concatenate((eta_,
                                      eta_high_))
        return eta_tilde_, eta_

    @partial(jit, static_argnums=[0, ])
    def _get_eta_tilde_and_fisher(self, theta, bijection_params):
        fisher = self._exponential_density.fisher_metric(theta, bijection_params)
        eta_extended = self._exponential_density.extended_statistics_expectation(theta, bijection_params)
        eta_tilde, _ = self.get_eta_tilde_and_eta_from_eta_extended_and_fisher(eta_extended, fisher)
        return eta_tilde, fisher
