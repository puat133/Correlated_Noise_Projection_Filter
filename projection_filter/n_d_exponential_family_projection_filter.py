from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as onp
import sympy as sp
from jax import jit
from jax.lax import scan

from projection_filter.exponential_family_projection_filter import ExponentialFamilyProjectionFilter
from projection_filter.n_d_exponential_family_halton import MultiDimensionalExponentialFamilyQMC
from projection_filter.n_d_exponential_family_hermite import MultiDimensionalExponentialFamilyHermite
from projection_filter.n_d_exponential_family_spg import MultiDimensionalExponentialFamilySPG
from symbolic.n_d import remove_monoms_from_remaining_stats, \
    construct_remaining_statistics
from symbolic.one_d import SDE
from symbolic.sympy_to_jax import sympy_matrix_to_jax
from utils.density_manipulations import create_2d_grid_from_limits


class MultiDimensionalProjectionFilter(ExponentialFamilyProjectionFilter, ABC):
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
                 epsilon: float = 1e-7,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray] = lambda x, params: jnp.arctanh(x),
                 ode_solver: str = 'euler',
                 sRule: str = "gauss-patterson",
                 bijection_parameters: tuple = None,
                 weight_cut_off: float = None,
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

        weight_cut_off: float
            weight cut off for the smolyak sparse grid

        Returns
        -------
        out : OneDimensionalSBulletProjectionFilter

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
        self._level = level
        self._nodes_number = nodes_number
        #       the statistics are assumed to be free of additional symbolic parameters
        self._natural_statistics, _ = \
            sympy_matrix_to_jax(
                self._natural_statistics_symbolic, self._dynamic_sde.variables)

        res = self._get_projection_filter_matrices()

        if res:# result can be None for the case MultiDimensionalCorrelatedProjectionFilter
            projection_filter_matrices_, monom_list_, remaining_monom_list_ = res
            #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
            self._projection_filter_matrices = projection_filter_matrices_
            self._monom_list = monom_list_
            self._remaining_monom_list = remaining_monom_list_
            # since some of monoms in remaining_monom_list_ can be expressed as c_i*c_j where c_i,c_j
            # are monoms from the natural statistics, we will remove these monoms, where we will
            # calculate their expectations from the fisher metric; E[c_ic_j] = I[i,j]+E[c_i]*E[c_j]
            self._higher_stats_indices_from_fisher, updated_remaining_monom_list = \
                remove_monoms_from_remaining_stats(self.natural_statistics_symbolic,
                                                   self._remaining_monom_list,
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
        else:
            self._extended_statistics_symbolic = self._natural_statistics_symbolic
            self._remaining_statistics = None

        self._integrator = integrator

        if integrator.lower() == 'spg':
            self._exponential_density = MultiDimensionalExponentialFamilySPG(
                sample_space_dimension=self._sample_space_dimension,
                sparse_grid_level=self._level,
                bijection=bijection,
                statistics=self._natural_statistics,
                remaining_statistics=self._remaining_statistics,
                epsilon=epsilon,
                sRule=sRule,
                bijection_parameters=bijection_parameters
            )
            self._integrator = integrator
        elif integrator.lower() == 'qmc':
            self._exponential_density = MultiDimensionalExponentialFamilyQMC(
                sample_space_dimension=self._sample_space_dimension,
                nodes_number=self._nodes_number,
                bijection=bijection,
                statistics=self._natural_statistics,
                remaining_statistics=self._remaining_statistics,
                bijection_parameters=bijection_parameters
            )
            self._integrator = integrator
        elif integrator.lower() == 'hermite':
            self._exponential_density = MultiDimensionalExponentialFamilyHermite(
                sample_space_dimension=self._sample_space_dimension,
                sparse_grid_level=self._level,
                statistics=self._natural_statistics,
                remaining_statistics=self._remaining_statistics,
                bijection_parameters=bijection_parameters,
                weight_cut_off=weight_cut_off
            )
            self._integrator = integrator
        else:
            raise ValueError('Integrator not recognized. At the moment, it should be either SPG for sparse grid,'
                             'HERMITE, for a sparse gauss-hermite grid, or QMC for quasi Monte Carlo')

        if initial_condition.shape[0] != self._exponential_density.params_num:
            raise Exception("Wrong initial condition shape!, expected {} "
                            "given {}".format(initial_condition.shape[0], self._exponential_density.params_num))
        self._current_state = initial_condition
        self._state_history = self._current_state[jnp.newaxis, :]

    @property
    def level(self):
        return self._level

    @property
    def nodes_number(self):
        return self.exponential_density.nodes_number

    @property
    def integrator_type(self):
        return self._integrator

    @property
    def monom_list(self):
        return self._monom_list

    @property
    def natural_statistics(self):
        return self._exponential_density.natural_statistics

    @property
    def remaining_statistics(self):
        return self._exponential_density.remaining_statistics

    @property
    def exponential_density(self):
        return self._exponential_density

    def get_density_values(self, grid_limits: jnp.ndarray, nb_of_points: jnp.ndarray):
        if grid_limits.ndim == 2:
            x_ = []
            for i in range(self._exponential_density.sample_space_dim):
                temp_ = jnp.linspace(
                    grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
                x_.append(temp_)
            grids = jnp.meshgrid(*x_, indexing='xy')
            grids = jnp.stack(grids, axis=-1)
            result = self.get_density_values_from_grids(grids)

        elif grid_limits.ndim == 3:
            quadrature_points = self._exponential_density.quadrature_points
            grid_history = create_2d_grid_from_limits(grid_limits, nb_of_points)

            @jit
            def _evaluate_density_loop_different_grids(carry_, input_):
                theta_, bijection_params_, grid_ = input_
                c_ = self.natural_statistics(grid_)
                psi_ = self._exponential_density.log_partition(theta_, bijection_params_)
                density_ = jnp.exp(c_ @ theta_ - psi_)
                bijected_points_ = self._exponential_density.bijection(quadrature_points, bijection_params_)
                return None, (density_, bijected_points_)

            _, (density_history_,
                bijection_points_history_
                ) = scan(_evaluate_density_loop_different_grids,
                         None, (self.state_history,
                                self.bijection_parameters_history,
                                grid_history))
            result = density_history_, bijection_points_history_

        return result

    def get_density_values_from_grids(self, grids):
        c = self.natural_statistics(grids)
        quadrature_points = self._exponential_density.quadrature_points

        @jit
        def _evaluate_density_loop(carry_, input_):
            theta_, bijection_params_ = input_
            psi_ = self._exponential_density.log_partition(theta_, bijection_params_)
            density_ = jnp.exp(c @ theta_ - psi_)
            bijected_points_ = self._exponential_density.bijection(quadrature_points, bijection_params_)
            return None, (density_, bijected_points_)

        _, (density_history_, bijection_points_history_) = scan(_evaluate_density_loop,
                                                                None, (self.state_history,
                                                                       self.bijection_parameters_history))
        return density_history_, bijection_points_history_

    def _construct_remaining_statistics(self):
        remaining_statistics_symbolic = construct_remaining_statistics(self._dynamic_sde,
                                                                       self._remaining_monom_list)
        return remaining_statistics_symbolic

    @abstractmethod
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
        return NotImplementedError

    @partial(jit, static_argnums=[0, ])
    def get_eta_tilde_and_eta_from_eta_extended_and_fisher(self, eta_extended_, fisher_):
        eta_ = eta_extended_[:self._exponential_density.params_num]
        eta_high_ = eta_extended_[self._exponential_density.params_num:]
        # expected value of additional natural statistics of the form c_i*c_j
        eta_combination_ = fisher_[self._higher_stats_indices_from_fisher] + \
                           eta_[self._higher_stats_indices_from_fisher[0]] * eta_[
                               self._higher_stats_indices_from_fisher[1]]
        eta_tilde_ = jnp.concatenate((eta_, eta_combination_, eta_high_))
        return eta_tilde_, eta_

    @partial(jit, static_argnums=[0, ])
    def _get_eta_tilde_and_fisher(self, theta, bijection_params):
        fisher = self._exponential_density.fisher_metric(theta, bijection_params)
        eta_extended = self._exponential_density.extended_statistics_expectation(theta, bijection_params)
        eta_tilde, _ = self.get_eta_tilde_and_eta_from_eta_extended_and_fisher(eta_extended, fisher)
        return eta_tilde, fisher
