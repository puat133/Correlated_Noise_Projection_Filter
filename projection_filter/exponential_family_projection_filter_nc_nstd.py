from abc import ABC
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sympy as sp
from jax import jit

from projection_filter.exponential_family_projection_filter import ExponentialFamilyProjectionFilter
from symbolic.one_d import SDE
from symbolic.n_d import get_projection_filter_statistics_correlated
from symbolic.sympy_to_jax import sympy_matrix_to_jax


class PlainProjectionFilter(ExponentialFamilyProjectionFilter, ABC):
    def __init__(self,
                 dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
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
        bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray]
        ode_solver: str
        bijection_parameters: tuple

        Returns
        -------
        out : PlainProjectionFilter

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
            bijection_parameters=bijection_parameters,
            rescale_measurement=False
        )

        len_dW = len(self._dynamic_sde.brownians)
        len_dV = len(self._measurement_sde.brownians)
        corr_mat = jnp.zeros((len_dW, len_dV))
        # compute required statistics symbolically
        self._required_statistics_symbolic = \
            get_projection_filter_statistics_correlated(natural_statistics_symbolic,
                                                        dynamic_sde,
                                                        measurement_sde,
                                                        corr_mat)

        # convert the symbolic statistics to jax functions
        # we do not need cal_F0,cal_F1,cal_F2,cal_F3,cal_F4 since they are 0
        self._required_statistics_symbolic = (self._required_statistics_symbolic[0],
                                              self._required_statistics_symbolic[1],
                                              self._required_statistics_symbolic[2],
                                              self._required_statistics_symbolic[3],
                                              self._required_statistics_symbolic[4],
                                              self._required_statistics_symbolic[5])

        self._all_symbolic_stats_flat = self._required_statistics_symbolic[0].vec()
        self._symbolic_stats_for_heun_update_flat = self._required_statistics_symbolic[0].vec()
        self._symbolic_stats_for_heun_update = [self._required_statistics_symbolic[0],
                                                self._required_statistics_symbolic[2],
                                                self._required_statistics_symbolic[4],
                                                ]  # h, hc, R
        for i, a_stat in enumerate(self._required_statistics_symbolic):
            if i != 0:
                self._all_symbolic_stats_flat = sp.Matrix([self._all_symbolic_stats_flat, a_stat.vec()])
                if i in (2, 4):
                    self._symbolic_stats_for_heun_update_flat = sp.Matrix(
                        [self._symbolic_stats_for_heun_update_flat, a_stat.vec()])
            if i > 5:
                break
        all_stat_funcs, _ = sympy_matrix_to_jax(self._all_symbolic_stats_flat, dynamic_sde.variables,
                                                disable_parameters=True)
        self._all_stats = jit(jnp.vectorize(all_stat_funcs, signature='(d)->(m)'))
        heun_funcs, _ = sympy_matrix_to_jax(self._symbolic_stats_for_heun_update_flat, dynamic_sde.variables,
                                            disable_parameters=True)
        self._stats_for_heun_update = jit(
            jnp.vectorize(heun_funcs, signature='(d)->(m)'))

        self._I = jnp.eye(self._measurement_sde.drifts.shape[0])

    @partial(jit, static_argnums=[0, ])
    def parameters_sde(self, theta, eta_tilde, fisher, dy, bijection_params):
        # TODO: the idea is to implement the parameters sde where all expected value from eq. 28 and hh^Tc and h^Tc are
        # calculated through the numerical integration. So we need to implement a method in exponential family class
        # such that for any function of x, s(x), expected value of s can be calculated for an exponential density
        # p_\theta, (theta is the natural parameter)

        # the required statistics are:
        # h, hh, hc, hhc, R, Ac, cal_F0, cal_F1, cal_F2, cal_F3, cal_F4
        ptheta = self.exponential_density

        start_index = 0
        e_all = ptheta.expected_value(self._all_stats, theta, bijection_params)

        e_matrices = []

        for i, a_stat in enumerate(self._required_statistics_symbolic):
            num_ell = len(a_stat)
            e_matrices.append(e_all[start_index:start_index + num_ell].reshape(a_stat.shape))
            start_index += num_ell
        eh, ehh, ehc, ehhc, eR, eAc = tuple(e_matrices)
        ec = eta_tilde[:self.exponential_density.params_num, jnp.newaxis]

        eN = ehc - jnp.outer(ec, eh)
        eP = ehhc - jnp.outer(ec, ehh.flatten())
        eR_inv = jnp.linalg.inv(eR)
        evec_R_inv = eR_inv.flatten()


        eZ = eP

        drift = jnp.linalg.solve(fisher, eAc.squeeze() - 0.5 * eZ @ evec_R_inv) * self._dt
        diffusion = jnp.linalg.solve(fisher, eN @ eR_inv @ dy)

        theta_aux = theta + diffusion

        bijection_params, eta_tilde_aux, fisher_aux = self.compute_eta_tilde_fisher_and_update_bijection_parameters(
            theta_aux,
            bijection_params)

        # Compute additional statistics after update
        start_index = 0
        e_all = ptheta.expected_value(self._stats_for_heun_update, theta_aux, bijection_params)

        e_matrices = []

        for i, a_stat in enumerate(self._symbolic_stats_for_heun_update):
            num_ell = len(a_stat)
            e_matrices.append(e_all[start_index:start_index + num_ell].reshape(a_stat.shape))
            start_index += num_ell
        eh_aux, ehc_aux, eR_aux = tuple(e_matrices)
        ec_aux = eta_tilde_aux[:self.exponential_density.params_num, jnp.newaxis]
        eN_aux = ehc_aux - jnp.outer(ec_aux, eh_aux)
        eR_inv_aux = jnp.linalg.inv(eR_aux)

        diffusion_aux = jnp.linalg.solve(fisher_aux, eN_aux @ eR_inv_aux @ dy)

        # Heun method for Stratonovich sde
        d_theta = drift + 0.5 * (diffusion + diffusion_aux)

        return d_theta, bijection_params
