import jax.numpy as jnp
import sympy as sp

from projection_filter.n_d_exponential_family_restricted_projection_filter_correlated import \
    MultiDimensionalRestrictedCorrelatedProjectionFilter
from projection_filter.n_d_exponential_family_projection_filter_gaussian_bijection import gauss_bijection
from symbolic.one_d import SDE


class MultiDimensionalCorrelatedFilterWithParametrizedBijection(MultiDimensionalRestrictedCorrelatedProjectionFilter):
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
                 ode_solver: str = 'euler',
                 sRule: str = "gauss-patterson",
                 bijection_parameters: tuple = None,
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
        dynamic_sde: SDE
            SDE for the dynamic.
        measurement_sde: SDE
            SDE for the measurement.
        natural_statistics_symbolic
        constants
        initial_condition
        measurement_record
        delta_t
        noise_correlation_matrix
        nodes_number
        integrator
        level
        epsilon
        ode_solver
        sRule
        bijection_parameters
        theta_indices_for_bijection_params
        moment_matching_iterations
        """
        super().__init__(dynamic_sde=dynamic_sde,
                         measurement_sde=measurement_sde,
                         natural_statistics_symbolic=natural_statistics_symbolic,
                         constants=constants,
                         initial_condition=initial_condition,
                         measurement_record=measurement_record,
                         delta_t=delta_t,
                         noise_correlation_matrix=noise_correlation_matrix,
                         nodes_number=nodes_number,
                         integrator=integrator,
                         level=level,
                         epsilon=epsilon,
                         bijection=gauss_bijection,
                         ode_solver=ode_solver,
                         sRule=sRule,
                         bijection_parameters=bijection_parameters,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params,
                         moment_matching_iterations=moment_matching_iterations
                         )


