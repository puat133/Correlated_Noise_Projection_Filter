from collections.abc import Callable
from functools import partial
from itertools import chain

import jax.numpy as jnp
import numpy as onp
import sympy as sp
from jax import jit
from jax.lax import scan

from projection_filter.exponential_family_projection_filter_correlated \
    import CorrelatedProjectionFilter
from projection_filter.n_d_exponential_family_projection_filter import MultiDimensionalProjectionFilter
from symbolic.one_d import SDE


class MultiDimensionalCorrelatedProjectionFilter(MultiDimensionalProjectionFilter,
                                                 CorrelatedProjectionFilter):

    def __init__(self,
                 dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 noise_correlation_matrix: jnp.ndarray,
                 nodes_number: int = 10,
                 integrator: str = 'spg',
                 level: int = 5,
                 epsilon: float = 1e-7,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray] = lambda x, params: jnp.arctanh(x),
                 ode_solver: str = 'euler',
                 sRule: str = "gauss-patterson",
                 bijection_parameters: tuple = None,
                 weight_cut_off: float = None,
                 theta_indices_for_bijection_params=jnp.array([0, 1], dtype=jnp.int32),
                 moment_matching_iterations: int = 1
                 ):
        """
        This abstract class inherits ExponentialFamilyProjectionFilter and
        encapsulates a class of projection filter for exponential family that satisfies
        the convenient choice given in Section 4 of Ref [1].

        Parameters
        ----------
        dynamic_sde: SDE
        measurement_sde: SDE
        natural_statistics_symbolic: sp.MutableDenseMatrix
        constants: dict
        initial_condition: jnp.ndarray
        measurement_record: jnp.ndarray
        delta_t: float
        noise_correlation_matrix: jnp.ndarray
        bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray]
        ode_solver: str
        bijection_parameters: tuple
        theta_indices_for_bijection_params: jnp.ndarray
        moment_matching_iterations: int


        Returns
        -------
        out : RestrictedCorrelatedProjectionFilter

        References
        ----------
        [1] M. F. Emzir, "Projection filter algorithm for correlated measurement and process noises,
        with state dependent covarianes"

        """

        # This need to be assigned first before calling the super
        self._S = noise_correlation_matrix
        self._theta_indices_for_bijection_params = theta_indices_for_bijection_params
        self._moment_matching_iterations: int = moment_matching_iterations

        CorrelatedProjectionFilter.__init__(self,
                                            dynamic_sde,
                                            measurement_sde,
                                            natural_statistics_symbolic,
                                            constants,
                                            initial_condition,
                                            measurement_record,
                                            delta_t,
                                            noise_correlation_matrix,
                                            bijection,
                                            ode_solver,
                                            bijection_parameters)

        # this will call MultidimensionalProjectionFilter __init__
        MultiDimensionalProjectionFilter.__init__(self,
                                                  dynamic_sde,
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
                                                  weight_cut_off,
                                                  rescale_measurement=False)

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
    def update_bijection_params(self, eta: jnp.array, fisher: jnp.array, old_bijection_params: tuple):
        _mu = jnp.take(eta, self._theta_indices_for_bijection_params[0])
        # _Sigma = fisher[self._indices_grid]
        _Sigma = eta[self._theta_indices_for_bijection_params[1]] - jnp.outer(_mu, _mu)
        _Sigma_eigvals, _Sigma_eigvects = jnp.linalg.eigh(_Sigma)

        # do not update scale factor
        _, _, _, scale_factor = old_bijection_params

        new_bijection_params = _mu, _Sigma_eigvals, _Sigma_eigvects, scale_factor
        return new_bijection_params

    @partial(jit, static_argnums=[0, ])
    def _fokker_planck(self, theta_: jnp.ndarray, bijection_params_: tuple, t: float):
        dtheta_dt = 0
        return dtheta_dt

    @property
    def projection_filter_matrices(self):
        return None

    def _get_projection_filter_matrices(self) -> tuple[list[onp.ndarray], list[tuple], list[tuple]]:
        return None

    @partial(jit, static_argnums=[0, ])
    def _get_eta_tilde_and_fisher(self, theta, bijection_params):
        fisher = self._exponential_density.fisher_metric(theta, bijection_params)
        eta_tilde = self._exponential_density.extended_statistics_expectation(theta, bijection_params)
        return eta_tilde, fisher
