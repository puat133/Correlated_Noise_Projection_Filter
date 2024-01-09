
import jax.numpy as jnp
import sympy as sp
from jax import jit
from jax.scipy.special import erfinv
from functools import partial
from projection_filter.n_d_exponential_family_projection_filter_s_star import MultiDimensionalSStarProjectionFilter
from symbolic.one_d import SDE


def plain_gauss_bijection(xtilde: jnp.ndarray, bijection_params: tuple):
    # since _Sigma_eigvects transpose is equal to its inverse
    _mu, _Sigma_eigvals, _Sigma_eigvects, scale_factor = bijection_params
    return _mu + scale_factor * jnp.sqrt(2) * _Sigma_eigvects.T@(jnp.sqrt(_Sigma_eigvals) * erfinv(xtilde))


gauss_bijection = jit(jnp.vectorize(plain_gauss_bijection, signature='(n)->(n)', excluded=(1,)))


class MultiDimensionalSStarProjectionFilterWithParametrizedBijection(MultiDimensionalSStarProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
                 measurement_sde: SDE,
                 natural_statistics_symbolic: sp.MutableDenseMatrix,
                 constants: dict,
                 initial_condition: jnp.ndarray,
                 measurement_record: jnp.ndarray,
                 delta_t: float,
                 bijection_parameters: tuple,
                 nodes_number: int = 1000,
                 integrator: str = 'spg',
                 level: int = 5,
                 epsilon: float = 0,
                 ode_solver: str = 'euler',
                 sRule: str = "gauss-patterson",
                 theta_indices_for_bijection_params=jnp.array([0, 1], dtype=jnp.int32),
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
                         integrator=integrator,
                         level=level,
                         epsilon=epsilon,
                         bijection=gauss_bijection,
                         ode_solver=ode_solver,
                         sRule=sRule,
                         bijection_parameters=bijection_parameters,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params,
                         moment_matching_iterations=moment_matching_iterations)

    @partial(jit, static_argnums=[0, ])
    def update_bijection_params(self, eta: jnp.array, fisher: jnp.array, old_bijection_params: tuple):
        return self.update_bijection_params_vanilla(eta,fisher,old_bijection_params)

