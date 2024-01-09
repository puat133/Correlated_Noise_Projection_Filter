from functools import partial
import jax.numpy as jnp
import sympy as sp
from jax import jit
from jax.scipy.special import erfinv
from jax.lax import scan
from projection_filter.one_d_exponential_family_projection_filter_s_star import OneDimensionalSStarProjectionFilter
from symbolic.one_d import SDE


def plain_gauss_bijection(xi: jnp.ndarray, bijection_params: tuple) -> jnp.ndarray:
    _mu, _Var, _scale_factor = bijection_params
    return _mu + _scale_factor * jnp.sqrt(2 * _Var) * erfinv(xi)


class OneDimensionalSStarProjectionFilterWithGaussBijection(OneDimensionalSStarProjectionFilter):
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
                 theta_indices_for_bijection_params= (jnp.array([0], dtype=jnp.int32),jnp.array([1], dtype=jnp.int32)),
                 moment_matching_iterations: int = 1):
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
        super().__init__(dynamic_sde=dynamic_sde,
                         measurement_sde=measurement_sde,
                         natural_statistics_symbolic=natural_statistics_symbolic,
                         constants=constants,
                         initial_condition=initial_condition,
                         measurement_record=measurement_record,
                         delta_t=delta_t,
                         nodes_number=nodes_number,
                         bijection=plain_gauss_bijection,
                         ode_solver=ode_solver,
                         bijection_parameters=bijection_parameters,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params,
                         moment_matching_iterations=moment_matching_iterations)

    @property
    def moment_matching_iterations(self):
        return self._moment_matching_iterations

    @moment_matching_iterations.setter
    def moment_matching_iterations(self, value: int):
        if value > 0:
            self._moment_matching_iterations = int(value)

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
        _Var = jnp.take(eta, self._theta_indices_for_bijection_params[1]) - _mu**2
        new_bijection_params = _mu[0], _Var[0], old_bijection_params[-1]
        return new_bijection_params
