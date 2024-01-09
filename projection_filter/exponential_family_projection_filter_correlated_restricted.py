from abc import ABC
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sympy as sp
from jax import jit

from projection_filter.exponential_family_projection_filter import ExponentialFamilyProjectionFilter
from symbolic.one_d import SDE


class RestrictedCorrelatedProjectionFilter(ExponentialFamilyProjectionFilter, ABC):
    def __init__(self,
                 dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 noise_correlation_matrix: jnp.ndarray,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray] = jnp.arctanh,
                 ode_solver: str = 'euler',
                 bijection_parameters: tuple = None):
        """
        This abstract class inherits ExponentialFamilyProjectionFilter and
        encapsulates a class of projection filter for exponential family that satisfies
        the convenient choice given in Section 4 of Ref [1].
        In this class, we assume that 1 is included in c_tilde

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

        Returns
        -------
        out : RestrictedCorrelatedProjectionFilter

        References
        ----------
        [1] M. F. Emzir, "Projection filter algorithm for correlated measurement and process noises,
        with state dependent covarianes"
        """
        super().__init__(
            dynamic_sde=dynamic_sde,
            measurement_sde=measurement_sde,
            natural_statistics_symbolic=natural_statistics_symbolic,
            constants=constants,
            initial_condition=initial_condition,
            measurement_record=measurement_record,
            delta_t=delta_t,
            bijection=bijection,
            ode_solver=ode_solver,
            bijection_parameters=bijection_parameters
        )

        # these lines below will never be executed by the child class, however, we put these lines so that we can use
        # these instance variables in _fokker_plank and parameters_sde method
        self._S = noise_correlation_matrix
        self._I = jnp.eye(measurement_record.shape[1])  # identity matrix of size m \times m
        self._F0 = None  # n_theta x m^2 x ntilde_theta
        self._F1 = None  # n_theta x m x ntilde_theta
        self._F2 = None  # m^2 x ntilde_theta
        self._F3 = None  # m n_theta x ntilde_theta
        self._F4 = None  # m^3 x ntilde_theta
        self._AR = None  # m^2 x ntilde_theta
        self._A = None  # n_theta x ntilde_theta
        self._H1 = None  # m x n_theta
        self._H2 = None  # m^2 x n_theta

    @partial(jit, static_argnums=[0, ])
    def _fokker_planck(self, theta_: jnp.ndarray, bijection_params_: tuple, t: float):
        dtheta_dt = 0
        return dtheta_dt

    @property
    def projection_filter_matrices(self):
        return self._F0, self._F1, self._F2, self._F3, self._F4, self._AR, self._A, self._H1, self._H2

    def _expect_val_m_plus_h(self, expect_r_inv: jnp.ndarray, eta_tilde: jnp.ndarray) -> jnp.ndarray:
        return (jnp.kron(self._I, expect_r_inv.flatten()) @ self._F4) @ eta_tilde

    def _expect_val_r_vect(self, eta_tilde) -> jnp.ndarray:
        return self._AR @ eta_tilde

    @partial(jit, static_argnums=[0, ])
    def parameters_sde(self, theta, eta_tilde, fisher, dy, bijection_params):
        ntheta = self._natural_statistics_symbolic.shape[0]
        m = self._measurement_sde.drifts.shape[0]

        # this should be solved using Stratonovich Euler Heun.
        eta = eta_tilde[:self.exponential_density.params_num]
        e_r_vect = self._expect_val_r_vect(eta_tilde)
        e_r_inv = jnp.linalg.inv(e_r_vect.reshape((self._dy.shape[1], self._dy.shape[1])))
        e_r_inv_vect = e_r_inv.flatten()
        e_m_plus_h = self._expect_val_m_plus_h(e_r_inv, eta_tilde)
        e_h = self._H1 @ eta
        e_m = e_m_plus_h - e_h

        diffusion = (self._H1.T
                     + jnp.linalg.solve(fisher,
                                        jnp.reshape(self._F3 @ eta_tilde, (ntheta, m), order='F'))) @ e_r_inv @ dy

        temp1 = self._A @ eta_tilde
        temp1 += -0.5 * ((self._F0 - jnp.kron((self._H1 @ eta)[jnp.newaxis, :, jnp.newaxis].T, self._F1))
                        @ eta_tilde) @ e_r_inv_vect
        drift = jnp.linalg.solve(fisher, temp1) * self._dt

        temp2 = jnp.kron((self._F1 @ eta_tilde), e_m_plus_h[jnp.newaxis, :]) - jnp.kron(eta[:, jnp.newaxis],
                                                                                        (self._F2 @ eta_tilde).T)
        temp2 = jnp.linalg.solve(fisher, temp2)
        temp2 += (jnp.kron(self._H1.T, e_m[jnp.newaxis, :]) + self._H2.T)
        temp2 = 0.5 * temp2 @ e_r_inv_vect

        drift -= temp2 * self._dt

        theta_aux = theta + diffusion

        bijection_params, eta_tilde_aux, fisher_aux = self.compute_eta_tilde_fisher_and_update_bijection_parameters(
            theta_aux,
            bijection_params)

        expected_val_r_vect_aux = self._expect_val_r_vect(eta_tilde_aux)
        expect_r_inv_aux = jnp.linalg.inv(expected_val_r_vect_aux.reshape((self._dy.shape[1], self._dy.shape[1])))

        diffusion_aux = (self._H1.T
                         + jnp.linalg.solve(fisher_aux,
                                            jnp.reshape(self._F3 @ eta_tilde_aux, (ntheta, m), order='F'))
                         ) @ expect_r_inv_aux @ dy

        d_theta = drift + 0.5 * (diffusion + diffusion_aux)

        return d_theta, bijection_params
