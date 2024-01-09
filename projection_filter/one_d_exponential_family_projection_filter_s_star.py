from inspect import getfullargspec
from collections.abc import Callable
import jax.numpy as jnp
import sympy as sp
from projection_filter.one_d_exponential_family import OneDExponentialFamily
from projection_filter.exponential_family_projection_filter_s_star import SStarProjectionFilter
from projection_filter.one_d_exponential_family_projection_filter import OneDExponentialFamilyProjectionFilter
from symbolic.one_d import SDE, column_polynomials_coefficients_one_D, \
    column_polynomials_maximum_degree_one_D, backward_diffusion
from symbolic.sympy_to_jax import lamdify
from symbolic.sympy_to_jax import sympy_matrix_to_jax


class OneDimensionalSStarProjectionFilter(OneDExponentialFamilyProjectionFilter, SStarProjectionFilter):
    def __init__(self, dynamic_sde: SDE,
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
                 theta_indices_for_bijection_params=(jnp.array([0], dtype=jnp.int32), jnp.array([1], dtype=jnp.int32)),
                 moment_matching_iterations: int = 1
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
        self._theta_indices_for_bijection_params = theta_indices_for_bijection_params
        self._moment_matching_iterations: int = moment_matching_iterations

        super(OneDimensionalSStarProjectionFilter, self).__init__(dynamic_sde,
                                                                  measurement_sde,
                                                                  natural_statistics_symbolic,
                                                                  constants,
                                                                  initial_condition,
                                                                  measurement_record,
                                                                  delta_t,
                                                                  nodes_number,
                                                                  bijection,
                                                                  ode_solver,
                                                                  bijection_parameters)

        #   since M_0, m_h, lamda starts with coefficient of x^0, then we need to remove the first column/entry
        _L = self._projection_filter_matrices[0]
        _A = self._projection_filter_matrices[1]
        _b = self._projection_filter_matrices[2]
        lamda = self._projection_filter_matrices[3]
        self._L_0 = _L[:, 1:]
        self._ell_0 = _L[:, 0]
        self._A_0 = _A[:, 1:]
        self._b_h = _b[0, 1:]
        self._a_0 = _A[:, 0]
        self._b_0 = _b[0, 0]
        self._lamda = lamda[:, 1:].T
        self._lambda_0 = lamda[:, 0]
        self._remaining_statistics_symbolic, self._higher_stats_indices_from_fisher = \
            self._construct_remaining_statistics()
        if len(self._remaining_statistics_symbolic)>0:
            self._extended_statistics_symbolic = sp.Matrix.vstack(self._natural_statistics_symbolic,
                                                                  self._remaining_statistics_symbolic)
            self._remaining_statistics, _ = \
                sympy_matrix_to_jax(self._remaining_statistics_symbolic, [self._dynamic_sde.variables[0], ])
        else:
            self._extended_statistics_symbolic = self._natural_statistics_symbolic
            self._remaining_statistics = None
        self._exponential_density = OneDExponentialFamily(nodes_number, bijection, self._natural_statistics,
                                                          self._remaining_statistics,
                                                          bijection_parameters=bijection_parameters)
        if initial_condition.shape[0] != self._exponential_density.params_num:
            raise Exception("Wrong initial condition shape!, expected {} "
                            "given {}".format(initial_condition.shape[0], self._exponential_density.params_num))

    def _get_projection_filter_matrices(self) -> tuple:
        """
        Get matrices related to a projection filter with an exponential family densities.

        Returns
        -------
        matrices: tuple
            list of matrices containing M_0, m_h, and lamda
        """
        Lc = backward_diffusion(self._natural_statistics_symbolic, self._dynamic_sde)

        #   squared measurement drift times natural statistics
        absh2c = self._natural_statistics_symbolic * self._measurement_sde.drifts[0] ** 2

        L_sym = column_polynomials_coefficients_one_D(Lc, self._dynamic_sde.variables[0])

        A_sym = column_polynomials_coefficients_one_D(Lc - absh2c / 2, self._dynamic_sde.variables[0])

        #   this would be the maximum degree in M_0, m_h, and lamda
        A_max_degree = column_polynomials_maximum_degree_one_D(Lc - absh2c / 2, self._dynamic_sde.variables[0])

        b_sym = column_polynomials_coefficients_one_D(sp.Matrix([self._measurement_sde.drifts[0] ** 2 / 2]),
                                                      self._dynamic_sde.variables[0])

        lamda_sym = column_polynomials_coefficients_one_D(sp.Matrix([self._measurement_sde.drifts[0]]),
                                                          self._dynamic_sde.variables[0])

        matrices = []
        expression_list = [L_sym, A_sym, b_sym, lamda_sym]

        #   this will convert the symbolic expression to ndarrays
        for i, expression in zip(range(len(expression_list)), expression_list):
            vector_fun = lamdify(expression)
            arguments_length = len(expression.free_symbols)
            if arguments_length == 0:
                matrix = jnp.array(vector_fun())
            else:
                full_arg_spec = getfullargspec(vector_fun)
                arg_names = full_arg_spec.args
                arguments_list = [self._constants[key] for key in arg_names]
                matrix = jnp.array(vector_fun(*arguments_list))  # there could be a better way to implement this.

            if (matrix.shape[1] != A_max_degree + 1) and i < 3:
                temp = jnp.pad(matrix, ((0, 0), (0, A_max_degree + 1 - matrix.shape[1])))
                matrix = temp

            # for lamda
            if i == 3 and (matrix.shape[1] < self._natural_statistics_symbolic.shape[0] + 1):
                temp = jnp.pad(matrix, ((0, 0), (0, self._natural_statistics_symbolic.shape[0] + 1 - matrix.shape[1])))
                matrix = temp

            matrices.append(matrix)

        return tuple(matrices)

    def _construct_remaining_statistics(self):
        natural_statistics_max_polynomial_degree = column_polynomials_maximum_degree_one_D(
            self._natural_statistics_symbolic,
            self._dynamic_sde.variables[0])
        M_0_max_degree_plus_one = self._A_0.shape[1] + 1
        remaining_statistic_list = []
        c = self._natural_statistics_symbolic
        n_theta = len(c)
        higher_stats_indices_from_fisher_list = []
        for i in range(natural_statistics_max_polynomial_degree + 1, M_0_max_degree_plus_one):
            remaining_stat = self._dynamic_sde.variables[0] ** i
            # finding a corresponding c_k and c_l so that remaining_stat = c_k*c_l
            for k in range(n_theta):
                for ell in range(n_theta):
                    if remaining_stat == c[k] * c[ell]:
                        higher_stats_indices_from_fisher_list.append((k, ell))
                        # Break the inner loop...
                        break
                else:
                    # Continue if the inner loop wasn't broken.
                    continue
                    # Inner loop was broken, break the outer.
                break

            # since we can get E[c_ic_j] from the fisher matrix and c_i and c_j is a monomial x^i
            # and x^j, where i,j <= natural_statistics_max_polynomial_degree, then
            if i > 2 * natural_statistics_max_polynomial_degree:
                remaining_statistic_list.append(remaining_stat)

        remaining_statistics_symbolic = sp.Matrix(remaining_statistic_list)
        temp = jnp.array(higher_stats_indices_from_fisher_list)
        higher_stats_indices_from_fisher = (temp[:, 0], temp[:, 1])
        return remaining_statistics_symbolic, higher_stats_indices_from_fisher

