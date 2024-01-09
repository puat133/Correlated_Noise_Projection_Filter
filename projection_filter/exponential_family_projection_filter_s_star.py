from abc import ABC
from functools import partial
from collections.abc import Callable

import jax.numpy as jnp
import sympy as sp
from jax import jit

from projection_filter.exponential_family_projection_filter import ExponentialFamilyProjectionFilter
from symbolic.one_d import SDE


class SStarProjectionFilter(ExponentialFamilyProjectionFilter, ABC):
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
        out : OneDimensionalSBulletProjectionFilter

        References
        ----------
        [1]

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
        self._L_0: jnp.ndarray = None
        self._ell_0: jnp.ndarray = None
        self._A_0: jnp.ndarray = None
        self._a_0: jnp.ndarray = None
        self._b_0: jnp.ndarray = None
        self._b_h: jnp.ndarray = None
        self._lamda: jnp.ndarray = None
        self._lambda_0: jnp.ndarray = None

    @partial(jit, static_argnums=[0, ])
    def _fokker_planck(self, theta_: jnp.ndarray, bijection_params_: tuple, t: float):
        fisher_ = self._exponential_density.fisher_metric(theta_, bijection_params_)
        eta_tilde_ = self._exponential_density.extended_statistics_expectation(theta_, bijection_params_)
        dtheta_dt = jnp.linalg.solve(fisher_, self._ell_0 + self._L_0 @ eta_tilde_)
        return dtheta_dt

    @property
    def projection_filter_matrices(self):
        return self._ell_0, self._L_0, self._a_0, self._A_0, self._b_h, self._lamda

    @partial(jit, static_argnums=[0, ])
    def parameters_sde(self, theta, eta_tilde, fisher, dy, bijection_params):
        eta = eta_tilde[:self._exponential_density.params_num]
        d_theta = jnp.linalg.solve(fisher, self._a_0 + self._b_0 * eta +
                                   (self._A_0 + jnp.outer(eta,
                                                          self._b_h)) @ eta_tilde) * self._dt + self._lamda @ dy
        return d_theta, bijection_params



