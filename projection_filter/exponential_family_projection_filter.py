from abc import ABC, abstractmethod
from functools import partial
from collections.abc import Callable

import jax.numpy as jnp
import numpy as onp
import sympy as sp
from jax import jit
from jax import lax
from jax.lax import scan

from projection_filter.exponential_family import ExponentialFamily
from symbolic.one_d import SDE


class ExponentialFamilyProjectionFilter(ABC):
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
                 bijection_parameters: tuple = None,
                 rescale_measurement: bool = True):
        """
        A Class that encapsulates the exponential family projection filter.
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
        out : OneDimensionalSBulletProjectionFilter

        References
        ----------
        [1]

        """

        self._dynamic_sde = dynamic_sde
        self._measurement_sde = measurement_sde
        self._natural_statistics_symbolic = natural_statistics_symbolic
        self._remaining_statistics_symbolic = None
        self._extended_statistics_symbolic = None
        self._constants = constants
        self._measurement_record = measurement_record
        self._dt = delta_t
        self._ode_solver = ode_solver
        self._dy = jnp.diff(self._measurement_record, axis=0)
        self._rescale_measurement = rescale_measurement
        if self._rescale_measurement:
            self._re_scale_measurement()

        if self._dy.ndim != 2:
            self._dy = self._dy[:, jnp.newaxis]
        self._t = jnp.arange(self._measurement_record.shape[0]) * self._dt
        self._sample_space_dimension = len(self._dynamic_sde.variables)
        self._current_state = initial_condition.copy()
        self._initial_condition = initial_condition
        self._state_history = self._current_state[jnp.newaxis, :]
        self._bijection_params_history = None
        self._bijection = bijection
        self._initial_bijection_params = bijection_parameters
        self._exponential_density: ExponentialFamily = None

        if self._ode_solver.lower() == 'rk':
            self._one_step_fokker_planck = self._runge_kutta
        elif self._ode_solver.lower() == 'euler':
            self._one_step_fokker_planck = self._euler
        else:
            self._ode_solver = 'euler'
            self._one_step_fokker_planck = self._euler

    def _re_scale_measurement(self):
        """
        The original projection filter is written for assumption that :math:`dV` is a standard Brownian

        .. math:: dy = h(x)dt + dV

        To Allow the measurement, with the same dV and non identity diffusion matrix :math:`G`

        .. math:: dy = h(x) dt + G dV

        then we need to use the scaled version: i.e

        .. math:: d \tilde{y} = \tilde{h} dt + \tilde{G} dV

        with :math:`\tilde{G}\tilde{G}^\top = I`. This is achieved by setting

        .. math:: d \tilde{y} = chol((GG^T)^{-1})
        Returns
        -------

        """
        g = self._measurement_sde.diffusions
        gg_T_inv = (g * g.transpose()).inv()
        gg_T_chol = gg_T_inv.cholesky()
        scaled_diffusion = gg_T_chol * g
        scaled_drift = gg_T_chol * self._measurement_sde.drifts
        scaled_measurement_SDE = SDE(drifts=scaled_drift, diffusions=scaled_diffusion, time=self._measurement_sde.time,
                                     variables=self._measurement_sde.variables,
                                     brownians=self._measurement_sde.brownians)
        gg_T_chol_np = jnp.atleast_2d(jnp.array(onp.array(gg_T_chol).astype(onp.float32)))
        if self._dy.ndim == 1:
            self._dy = gg_T_chol_np[0, 0] * self._dy
        elif self._dy.ndim == 2:
            self._dy = gg_T_chol_np[jnp.newaxis, :, :] @ self._dy[:, :, jnp.newaxis]
        self._dy = self._dy.squeeze()
        self._measurement_sde = scaled_measurement_SDE

    @abstractmethod
    def _fokker_planck(self, theta_: jnp.ndarray, bijection_params_: tuple, t: float):
        return NotImplementedError

    @partial(jit, static_argnums=[0, ])
    def _runge_kutta(self, theta_: jnp.ndarray, bijection_params_: tuple, t: float):
        k1 = self._fokker_planck(theta_, bijection_params_, t)
        k2 = self._fokker_planck(theta_ + 0.5 * self._dt * k1, bijection_params_, t + 0.5 * self._dt)
        k3 = self._fokker_planck(theta_ + 0.5 * self._dt * k2, bijection_params_, t + 0.5 * self._dt)
        k4 = self._fokker_planck(theta_ + self._dt * k3, bijection_params_, t + self._dt)
        new_theta = theta_ + self._dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return new_theta

    @partial(jit, static_argnums=[0, ])
    def _euler(self, theta_: jnp.ndarray, bijection_params_: tuple, t: float):
        return theta_ + self._dt * self._fokker_planck(theta_, bijection_params_, t)

    @property
    @abstractmethod
    def projection_filter_matrices(self):
        return NotImplementedError

    @property
    def ode_solver(self):
        return self._ode_solver

    @ode_solver.setter
    def ode_solver(self, value):
        if value.lower() == 'rk':
            self._ode_solver = value
            self._one_step_fokker_planck = self._runge_kutta
        elif value.lower() == 'euler':
            self._ode_solver = value
            self._one_step_fokker_planck = self._euler

    @property
    def natural_statistics_symbolic(self):
        return self._natural_statistics_symbolic

    @property
    def extended_statistics_symbolic(self):
        return self._extended_statistics_symbolic

    @property
    @abstractmethod
    def natural_statistics(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def remaining_statistics(self):
        raise NotImplementedError

    @property
    def current_state(self):
        return self._current_state

    @property
    def state_history(self):
        return self._state_history

    @property
    def bijection_parameters_history(self):
        return self._bijection_params_history

    @property
    @abstractmethod
    def exponential_density(self):
        raise NotImplementedError

    def get_natural_statistics_expectation(self):

        @jit
        def _get_natural_statistics_expectation(carry, inputs):
            _state, _bijection_params = inputs
            a_moment = self.exponential_density.natural_statistics_expectation(_state,
                                                                               _bijection_params)
            return None, a_moment

        _, expected_natural_statistics_history = scan(_get_natural_statistics_expectation, None,
                                                      (self.state_history, self.bijection_parameters_history)
                                                      )
        return expected_natural_statistics_history

    def get_extended_statistics_expectation(self):
        @jit
        def _get_extended_statistics_expectation(carry, inputs):
            _state, _bijection_params = inputs
            a_moment = self.exponential_density.extended_statistics_expectation(_state,
                                                                                _bijection_params)
            return None, a_moment

        _, expected_extended_natural_statistics_history = scan(_get_extended_statistics_expectation, None,
                                                               (self.state_history,
                                                                self.bijection_parameters_history)
                                                               )
        return expected_extended_natural_statistics_history

    @abstractmethod
    def get_density_values(self, grid_limits: jnp.ndarray, nb_of_points: jnp.ndarray):
        """
        get density values for given `grid_limits` as an array of `N_d x 2` and `nb_of_points` as an
        array of `N_d` integers
        Parameters
        ----------
        grid_limits
        nb_of_points

        Returns
        -------

        """
        raise NotImplementedError

    @partial(jit, static_argnums=[0, ])
    def update_bijection_params(self, eta: jnp.array, fisher: jnp.array, old_bijection_params: tuple):
        return old_bijection_params

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
        fisher = self._exponential_density.fisher_metric(theta, bijection_params)
        eta_tilde = self._exponential_density.extended_statistics_expectation(theta, bijection_params)
        return bijection_params, eta_tilde, fisher

    def propagate(self):
        self._propagate_filter()

    def solve_Fokker_Planck(self):
        self._propagate_FK()

    def _propagate_FK(self):
        @jit
        def integrator_loop(carry_, inputs_):
            t_ = inputs_
            theta_, bijection_params_ = carry_

            theta_ = self._one_step_fokker_planck(theta_, bijection_params_, t_)

            # update bijection parameters after getting new theta
            bijection_params_ = self.compute_eta_tilde_fisher_and_update_bijection_parameters(theta_, bijection_params_)

            carry_ = theta_, bijection_params_
            return carry_, carry_

        _carry, _history = lax.scan(integrator_loop, (self._current_state, self._exponential_density.bijection_params)
                                    , self._t[1:])
        self._current_state, self._exponential_density.bijection_params = _carry
        self._state_history, self._bijection_params_history = _history

    def _propagate_filter(self):

        @jit
        def integrator_loop(carry_, inputs_):
            t_, dy_ = inputs_
            theta_, bijection_params_, eta_tilde_, fisher_ = carry_
            d_theta_, bijection_params_ = self.parameters_sde(theta_, eta_tilde_, fisher_, dy_, bijection_params_)
            theta_ = theta_ + d_theta_
            # this needs to be done since to compute the updated bijection_params, we need to evaluate
            # new eta and fisher matrix according to the new theta
            bijection_params_, eta_tilde_, fisher_ = self.compute_eta_tilde_fisher_and_update_bijection_parameters(
                theta_,
                bijection_params_)

            carry_ = theta_, bijection_params_, eta_tilde_, fisher_

            return carry_, (theta_, bijection_params_)

        bijection_params = self._exponential_density.bijection_params
        theta = self._current_state
        eta_tilde, fisher = self._get_eta_tilde_and_fisher(theta, bijection_params)
        _carry, _history = lax.scan(integrator_loop, (theta, bijection_params, eta_tilde, fisher)
                                    , [self._t[1:], self._dy])
        self._current_state, self._exponential_density.bijection_params, _, _ = _carry
        self._state_history, self._bijection_params_history = _history
        self._append_initial_condition()

    @abstractmethod
    def parameters_sde(self, theta, eta_tilde, fisher, dy, bijection_params):
        NotImplementedError
    def _append_initial_condition(self):
        # append initial condition to state_history
        self._state_history = jnp.insert(self._state_history, 0, self._initial_condition, axis=0)

        # append initial bijection parameter to bijection_params_history
        modified_bijection_param_history = []
        if self._bijection_params_history:
            for i in range(len(self._bijection_params_history)):
                param_history = self._bijection_params_history[i]
                param_history = jnp.insert(param_history, 0, self._initial_bijection_params[i], axis=0)
                modified_bijection_param_history.append(param_history)

        self._bijection_params_history = tuple(modified_bijection_param_history)

    def discrete_propagate(self):
        """
        This is used in the case that the measurement is a discrete process
        Returns
        -------

        """
        self._propagate_filter_two_step()

    def _propagate_filter_two_step(self):
        @jit
        def integrator_loop(carry_, inputs_):
            t_, dy_ = inputs_
            theta_ = carry_

            # predictive update
            theta_ = self._one_step_fokker_planck(theta_, t_)
            # bayesian update
            theta_ += self._lamda @ dy_ - self._lambda_0

            carry_ = theta_
            return carry_, carry_

        self._current_state, _history = lax.scan(integrator_loop, self._current_state, [self._t[1:],
                                                                                        self._dy])
        self._state_history = _history

    @abstractmethod
    def _construct_remaining_statistics(self):
        raise NotImplementedError

    @abstractmethod
    def _get_projection_filter_matrices(self):
        """
        Get matrices related to a projection filter with an exponential family densities.

        Returns
        -------
        matrices: tuple
            list of matrices related to projection filter
        """
        raise NotImplementedError

    def empirical_kld(self,
                      samples_history:jnp.ndarray)-> float:
        """
        Relative entropy from an empirical density with samples, to
        an exponential density with natural parameters theta and bijection_params.
        It is assumed that the samples has uniform weights.

        Parameters
        ----------
        samples_history: jnp.ndarray
            samples history from an empirical density



        Returns
        -------
        result: float
            relative entropy.

        """
        p = self._exponential_density
        N = self._state_history.shape[1] #sample size
        mean_log_weights = -jnp.log(N)/N
        def _relative_entropy(carry, inputs):
            samples, theta, bijection_params = inputs
            psi = p.log_partition(theta, bijection_params)
            relative_entropy = mean_log_weights - jnp.mean((p.natural_statistics(samples)@ theta - psi))
            return carry, relative_entropy

        _, rel_ent_history = lax.scan(_relative_entropy,None, (samples_history,
                                                               self._state_history,
                                                               self._bijection_params_history))

        return rel_ent_history

    @partial(jit, static_argnums=[0, ])
    def _get_eta_tilde_and_fisher(self, theta, bijection_params):
        fisher = self._exponential_density.fisher_metric(theta, bijection_params)
        eta_tilde = self._exponential_density.extended_statistics_expectation(theta, bijection_params)
        return eta_tilde, fisher
