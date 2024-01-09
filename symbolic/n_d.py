import sympy as sp
import numpy as onp
import jax.numpy as jnp
from itertools import chain
from symbolic.one_d import SDE


def mix_monomials_up_to_order(x: tuple, max_order: int, norm_order: int = 1) -> sp.MutableDenseMatrix:
    """
    Generate a matrix containing monomials with order up to max_order

    Parameters
    ----------
    norm_order: int
        norm order to decide the monomial
    x: tuple
        tuple of symbols x1,...,xd
    max_order: int
        maximum order

    Returns
    -------
    monomials: sp.Matrix
        the monomial excluding a constant 1.

    """
    dim = len(x)
    order = onp.arange(max_order + 1)
    orders = onp.stack(onp.meshgrid(*onp.tile(order, (dim, 1))), axis=-1)
    orders_lined = orders.reshape(-1, dim)
    selected_order = orders_lined[onp.linalg.norm(orders_lined, ord=norm_order,
                                                  axis=-1) < max_order + 1]
    selected_order_list = list(map(tuple, selected_order))  # convert to list of tuple
    selected_order_list.sort()
    monomials = []
    x_ = onp.array(x)
    for i in range(1, selected_order.shape[0]):
        monomials.append(sp.prod(onp.power(x_, selected_order_list[i])))
    return sp.Matrix(monomials)


def hyperbolic_cross_monomials(x: tuple, max_order: int) -> sp.MutableDenseMatrix:
    """
    Generate a matrix containing monomials with order up to max_order

    Parameters
    ----------
    x: tuple
        tuple of symbols x1,...,xd
    max_order: int
        maximum order

    Returns
    -------
    monomials: sp.Matrix
        the monomial excluding a constant 1.

    """
    dim = len(x)
    order = onp.arange(max_order + 1)
    orders = onp.stack(onp.meshgrid(*onp.tile(order, (dim, 1))), axis=-1)
    max_1_orders = onp.maximum(1, orders)
    max_1_orders_lined = max_1_orders.reshape(-1, dim)
    orders_lined = orders.reshape(-1, dim)
    selected_order = orders_lined[onp.prod(max_1_orders_lined, axis=-1) < max_order + 1]
    selected_order_list = list(map(tuple, selected_order))  # convert to list of tuple
    selected_order_list.sort()  # convert to list of tuple
    monomials = []
    x_ = onp.array(x)
    for i in range(1, selected_order.shape[0]):
        monomials.append(sp.prod(onp.power(x_, selected_order_list[i])))
    return sp.Matrix(monomials)


def backward_diffusion(fun_array: sp.MutableDenseMatrix, sde: SDE) \
        -> sp.MutableDenseMatrix:
    """
    Compute backward diffusion operator of a given function for a given sde. The function and sde
    should be function of the same symbolic variable.

    Parameters
    ----------
    fun_array : Callable[[sympy.Symbol], sympy.Symbol]
    sde : SDE

    Returns
    -------
    res : sympy.matrices.dense.MutableDenseMatrix
    """
    jac = fun_array.jacobian(sde.variables)
    res = jac * sde.drifts
    for i in range(jac.shape[0]):
        res[i] += sp.trace(sde.diffusions.transpose() * jac[i, :].jacobian(sde.variables) * sde.diffusions) / 2
    return res


def get_monomial_degree_set(fun_array: sp.MutableDenseMatrix, variables: tuple[sp.Symbol]) -> set[tuple[int]]:
    """
    Get coefficients of monomials from an array of polynomials given in terms of variables in a tuple.

    Parameters
    ----------
    fun_array
    variables

    Returns
    -------

    """
    monomial_degree_set = set()
    for an_entry in fun_array:
        monoms = an_entry.as_poly(variables).monoms()
        for monom in monoms:
            monomial_degree_set.add(monom)

    # res = list(monomial_degree_set)
    # res.sort()
    return monomial_degree_set


def from_tuple_to_symbolic_monom(variables: tuple[sp.Symbol], degree_tuple: tuple[int]) -> sp.Symbol:
    """
    Convert

    Parameters
    ----------
    variables
    degree_tuple

    Returns
    -------

    """
    res = 1
    for i in range(len(degree_tuple)):
        res *= variables[i] ** degree_tuple[i]
    return res


def column_polynomials_coefficients(col: sp.MutableDenseMatrix, variables: tuple[sp.Symbol],
                                    monomial_degree_list: list[tuple[int, int]] = None) -> \
        tuple[list[sp.Symbol], onp.ndarray]:
    """

    Parameters
    ----------
    monomial_degree_list
    col
    variables

    Returns
    -------

    """
    if not monomial_degree_list:
        monomial_degree_list = get_monomial_degree_set(col, variables)

    monoms = [from_tuple_to_symbolic_monom(variables, monomial_degree) for monomial_degree in monomial_degree_list]
    coefficients = onp.zeros((col.shape[0], len(monoms)))
    for i in range(col.shape[0]):
        for j in range(len(monoms)):
            coefficients[i, j] = col[i].as_poly(variables).coeff_monomial(monoms[j])

    return monoms, coefficients


def compute_f_2_and_f_4(x: sp.Matrix,
                        h: sp.Matrix,
                        R: sp.Matrix,
                        gamma: sp.Matrix):
    n = x.shape[0]
    cal_F2 = sp.Matrix(jnp.zeros((h.shape[0] * gamma.shape[1],), dtype=jnp.int32))
    vect_R = sp.Matrix(R.flat())
    cal_F4 = sp.kronecker_product(vect_R, h)
    for i in range(n):
        Gi = gamma[i, :].transpose()
        cal_F2 += sp.kronecker_product(sp.diff(h, x[i]), Gi)
        cal_F4 += sp.kronecker_product(sp.diff(vect_R, x[i]), Gi)

    return cal_F2, cal_F4


# since m = 1, then kronecker product equal to just ordinary multiplication
def compute_f_0_1_3(x: sp.Matrix,
                    h: sp.Matrix,
                    natural_statistic: sp.Matrix,
                    gamma: sp.Matrix):
    """
    Calculate the F_0,F_1,F_2,F_3, and F_4 from Ref[1] Section 4.

    Parameters
    ----------
    x
    h
    R
    natural_statistic
    gamma

    Returns
    -------
    res: sp.Matrix

    References
        ----------
        [1] M. F. Emzir, "Projection filter algorithm for correlated measurement and process noises,
        with state dependent covarianes"
    """
    n = x.shape[0]
    cal_F0 = sp.Matrix(jnp.zeros((h.shape[0] * gamma.shape[1],), dtype=jnp.int32))
    cal_F1 = sp.Matrix(jnp.zeros((gamma.shape[1],), dtype=jnp.int32))
    cal_F3 = sp.Matrix(jnp.zeros((gamma.shape[1],), dtype=jnp.int32))

    for i in range(n):
        Gi = gamma[i, :].transpose()
        cal_F0 += (sp.kronecker_product(Gi, h) + sp.kronecker_product(h, Gi)) * sp.diff(natural_statistic, x[i])
        cal_F0 += sp.kronecker_product(sp.diff(h, x[i]), Gi) * natural_statistic
        cal_F1 += Gi * sp.diff(natural_statistic, x[i])
        cal_F3 += sp.kronecker_product(Gi, sp.diff(natural_statistic, x[i]))
        for j in range(n):
            Gj = gamma[j, :].transpose()
            cal_F0 += sp.kronecker_product(sp.diff(Gi, x[j]), Gj) * sp.diff(natural_statistic, x[i])
            cal_F0 += sp.kronecker_product(Gi, Gj) * sp.diff(sp.diff(natural_statistic, x[i]), x[j])

    return cal_F0, cal_F1, cal_F3


def compute_capital_f_of_statistics(x: sp.Matrix,
                                    h: sp.Matrix,
                                    R: sp.Matrix,
                                    natural_statistics: sp.Matrix,
                                    gamma: sp.Matrix):
    """

    Parameters
    ----------
    x
    h
    R
    natural_statistics
    gamma

    Returns
    -------

    """

    cal_F2, cal_F4 = compute_f_2_and_f_4(x, h, R, gamma)
    cal_F0_list = []
    cal_F1_list = []
    cal_F3_list = []
    for i in range(len(natural_statistics)):
        c = natural_statistics[i, :]
        cal_F0_, cal_F1_, cal_F3_ = compute_f_0_1_3(x, h, c, gamma)
        cal_F0_list.append([cal_F0_.T])
        cal_F1_list.append([cal_F1_.T])
        cal_F3_list.append([cal_F3_.T])

    cal_F0 = sp.Matrix(cal_F0_list)
    cal_F1 = sp.Matrix(cal_F1_list)
    cal_F3 = sp.Matrix(cal_F3_list)

    return cal_F0, cal_F1, cal_F2, cal_F3, cal_F4


def get_projection_filter_statistics_correlated(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                                dynamic_sde: SDE,
                                                measurement_sde: SDE,
                                                cross_covariance_matrix: jnp.ndarray,
                                                simplified=False
                                                ):
    """

    Parameters
    ----------
    natural_statistics_symbolic
    dynamic_sde
    measurement_sde
    cross_covariance_matrix

    Returns
    -------
    tuples of vector functions of dynamic_sde.variables:
        h, hh, hhc, Ac, R, gamma, F0, F1, F2, F3, F4
    """
    x = dynamic_sde.variables
    c = natural_statistics_symbolic
    Ac = backward_diffusion(natural_statistics_symbolic, dynamic_sde)
    h = measurement_sde.drifts
    hc = c * h.transpose()
    hh = h * h.transpose()
    hhc = c * (sp.kronecker_product(h, h)).transpose()
    R = measurement_sde.diffusions * measurement_sde.diffusions.transpose()
    S = sp.Matrix(cross_covariance_matrix.tolist())
    gamma = dynamic_sde.diffusions * S * measurement_sde.diffusions.transpose()
    cal_F0, cal_F1, cal_F2, cal_F3, cal_F4 = compute_capital_f_of_statistics(sp.Matrix(list(x)), h, R, c, gamma)

    if not simplified:
        return h, hh, hc, hhc, R, Ac, cal_F0, cal_F1, cal_F2, cal_F3, cal_F4
    else:
        return sp.simplify(h), sp.simplify(hh), sp.simplify(hc), sp.simplify(hhc), sp.simplify(R), sp.simplify(Ac), \
            sp.simplify(cal_F0), sp.simplify(cal_F1), sp.simplify(cal_F2), sp.simplify(cal_F3), sp.simplify(cal_F4)


def get_projection_filter_matrices_correlated(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                              dynamic_sde: SDE,
                                              measurement_sde: SDE,
                                              cross_covariance_matrix: jnp.ndarray,
                                              ) -> tuple[list[onp.ndarray], list[tuple], list[tuple]]:
    """

    Parameters
    ----------
    natural_statistics_symbolic
    dynamic_sde
    measurement_sde
    cross_covariance_matrix

    Returns
    -------

    """
    x = dynamic_sde.variables
    c = natural_statistics_symbolic
    h, hh, hc, hhc, R, Ac, cal_F0, cal_F1, cal_F2, cal_F3, cal_F4 = \
        get_projection_filter_statistics_correlated(natural_statistics_symbolic,
                                                    dynamic_sde,
                                                    measurement_sde,
                                                    cross_covariance_matrix)

    natural_monom_set = get_monomial_degree_set(c, x)
    monom_set = natural_monom_set.union(get_monomial_degree_set(Ac, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F0, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F1, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F2, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F3, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F4, x))
    remaining_monoms_set = monom_set.difference(natural_monom_set)

    constant_monom = tuple([0 for x_ in x])
    if constant_monom in remaining_monoms_set:
        remaining_monoms_set.remove(constant_monom)
        # constant is removed from remaining monoms_set temporarily but will be put back at its end

    natural_monom_list = list(natural_monom_set)
    natural_monom_list.sort()
    remaining_monoms_list = list(remaining_monoms_set)
    remaining_monoms_list.sort()
    remaining_monoms_list.append(constant_monom)  # we put back the constant monom here
    monom_list = list(chain.from_iterable(
        [natural_monom_list, remaining_monoms_list]))

    _, A = column_polynomials_coefficients(Ac.vec(), x, monom_list)

    F0 = []
    F1 = []
    for i in range(len(c)):
        _, F0_ = column_polynomials_coefficients(cal_F0[i, :].vec(), x, monom_list)
        _, F1_ = column_polynomials_coefficients(cal_F1[i, :].vec(), x, monom_list)

        F0.append(F0_)
        F1.append(F1_)

    F0 = jnp.stack(F0)
    F1 = jnp.stack(F1)

    _, F2 = column_polynomials_coefficients(cal_F2.vec(), x, monom_list)
    _, F3 = column_polynomials_coefficients(cal_F3.vec(), x, monom_list)
    _, F4 = column_polynomials_coefficients(cal_F4.vec(), x, monom_list)
    _, AR = column_polynomials_coefficients(R.vec(), x, monom_list)
    _, H1 = column_polynomials_coefficients(h.vec(), x, natural_monom_list)
    _, H2 = column_polynomials_coefficients(hh.vec(), x, natural_monom_list)

    return [F0, F1, F2, F3, F4, AR, A, H1, H2], monom_list, remaining_monoms_list


def remove_monoms_from_remaining_stats(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                       remaining_monom_list: list,
                                       dynamic_sde: SDE
                                       ):
    """
    since some of monoms in remaining_monom_list_ can be expressed as c_i*c_j where c_i,c_j
    are monoms from the natural statistics, we will remove these monoms, where we will
    calculate their expectations from the fisher metric; E[c_ic_j] = I[i,j]+E[c_i]*E[c_j]
    Returns
    -------
    higher_stats_indices_from_fisher_list, monoms_tuples_to_be_removed_list

    """
    higher_stats_indices_from_fisher_list = []
    monoms_tuples_to_be_removed_list = []
    c = natural_statistics_symbolic
    n_theta = len(c)
    for a_monom_degree in remaining_monom_list:
        a_monom = from_tuple_to_symbolic_monom(dynamic_sde.variables, a_monom_degree)
        for k in range(n_theta):
            for ell in range(k, n_theta):
                if a_monom == c[k] * c[ell]:
                    higher_stats_indices_from_fisher_list.append((k, ell))
                    monoms_tuples_to_be_removed_list.append(a_monom_degree)
                    # Break the inner loop...
                    break
            else:
                # Continue if the inner loop wasn't broken.
                continue
                # Inner loop was broken, break the outer.
            break

    # now that we have collected them, remove them from remaining_monom_list
    for a_monom_degree in monoms_tuples_to_be_removed_list:
        remaining_monom_list.remove(a_monom_degree)

    if higher_stats_indices_from_fisher_list:
        temp = jnp.array(higher_stats_indices_from_fisher_list)
        higher_stats_indices_from_fisher = (temp[:, 0], temp[:, 1])
    else:
        higher_stats_indices_from_fisher = ([], [])
    updated_remaining_monom_list = remaining_monom_list

    return higher_stats_indices_from_fisher, updated_remaining_monom_list


def construct_remaining_statistics(dynamic_sde: SDE,
                                   remaining_monom_list: list
                                   ):
    """

    Parameters
    ----------
    dynamic_sde
    remaining_monom_list

    Returns
    -------

    """
    remaining_monoms = [from_tuple_to_symbolic_monom(dynamic_sde.variables, monomial_degree)
                        for monomial_degree in remaining_monom_list]
    remaining_statistics_symbolic = sp.Matrix(remaining_monoms)
    return remaining_statistics_symbolic
