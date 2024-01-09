import numpy as np
import numba as nb
from .smolyak import OneDNodesAndWeights, smolyak_grid


@nb.njit(fastmath=True)
def abwe1(n: int, m: int, tol: float, coef2: float, even: bool, b: np.ndarray, x: float) -> tuple[float, float]:
    """
    ABWE1 calculates a Kronrod abscissa and weight.

    Parameters
    ----------
    n : int
        the order of the Gauss rule.
    m : int
        the value of ( n + 1 ) / 2.
    tol : float
        the requested absolute accuracy of the abscissas.
    coef2 : float
        a value needed to compute weights.
    even: bool
        whether it is even or not
    b   : np.ndarray
        the Chebyshev coefficients.
    x   : float
        an estimate for the abscissa.

    Returns
    -------
    res : tuple[float,float]
        tuple contains  the abscissa and the weight

    References
    ----------
    https://people.math.sc.edu/Burkardt/py_src/kronrod/kronrod.py
    """

    if x == 0.0:
        ka = 1
    else:
        ka = 0
    #
    #  Iterative process for the computation of a Kronrod abscissa.
    #
    for _iter in range(1, 51):
        b1 = 0.0
        b2 = b[m + 1 - 1]
        yy = 4.0 * x * x - 2.0
        d1 = 0.0

        if even:
            ai = m + m + 1
            d2 = ai * b[m + 1 - 1]
            dif = 2.0
        else:
            ai = m + 1
            d2 = 0.0
            dif = 1.0

        for k in range(1, m + 1):
            ai = ai - dif
            i = m - k + 1
            b0 = b1
            b1 = b2
            d0 = d1
            d1 = d2
            b2 = yy * b1 - b0 + b[i - 1]
            if not even:
                i = i + 1
            d2 = yy * d1 - d0 + ai * b[i - 1]

        if even:
            f = x * (b2 - b1)
            fd = d2 + d1
        else:
            f = 0.5 * (b2 - b0)
            fd = 4.0 * x * d2
        #
        #  Newton correction.
        #
        delta = f / fd
        x = x - delta

        if ka == 1:
            break

        if abs(delta) <= tol:
            ka = 1
    #
    #  Catch non-convergence.
    #
    if ka != 1:
        raise Exception('ABWE1 - Fatal error!')
        # print('')
        # print('ABWE1 - Fatal error!')
        # print('  Iteration limit reached.')
        # print('  Last DELTA was %e' % (delta))
        # exit('ABWE1 - Fatal error!')
    #
    #  Computation of the weight.
    #
    d0 = 1.0
    d1 = x
    ai = 0.0
    for k in range(2, n + 1):
        ai = ai + 1.0
        d2 = ((ai + ai + 1.0) * x * d1 - ai * d0) / (ai + 1.0)
        d0 = d1
        d1 = d2

    w = coef2 / (fd * d2)

    return x, w


@nb.njit(fastmath=True)
def abwe2(n: int, m: int, tol: float, coef2: float, even: bool, b: np.ndarray, x: float) -> tuple[float, float, float]:
    """
    ABWE2 calculates a Gaussian abscissa and two weights.

     Parameters
    ----------
    n : int
        the order of the Gauss rule.
    m : int
        the value of ( n + 1 ) / 2.
    tol : float
        the requested absolute accuracy of the abscissas.
    coef2 : float
        a value needed to compute weights.
    even: bool
        whether it is even or not
    b   : np.ndarray
        the Chebyshev coefficients.
    x   : float
        an estimate for the abscissa.

    Returns
    -------
    res : tuple[float,float, float]
        tuple contains  the abscissa, the Gauss-Kronrod weight, and the Gauss weight.

    References
    ----------
    https://people.math.sc.edu/Burkardt/py_src/kronrod/kronrod.py

    """
    if x == 0.0:
        ka = 1
    else:
        ka = 0
        #
        #  Iterative process for the computation of a Gaussian abscissa.
        #
    for _iter in range(1, 51):

        p0 = 1.0
        p1 = x
        pd0 = 0.0
        pd1 = 1.0
        #
        #  When N is 1, we need to initialize P2 and PD2 to avoid problems with DELTA.
        #
        if n <= 1:
            if x != 0.0:
                p2 = (3.0 * x * x - 1.0) / 2.0
                pd2 = 3.0 * x
            else:
                p2 = 3.0 * x
                pd2 = 3.0

        ai = 0.0
        for k in range(2, n + 1):
            ai = ai + 1.0
            p2 = ((ai + ai + 1.0) * x * p1 - ai * p0) / (ai + 1.0)
            pd2 = ((ai + ai + 1.0) * (p1 + x * pd1) - ai * pd0) / (ai + 1.0)
            p0 = p1
            p1 = p2
            pd0 = pd1
            pd1 = pd2
        #
        #  Newton correction.
        #
        delta = p2 / pd2
        x = x - delta

        if ka == 1:
            break

        if abs(delta) <= tol:
            ka = 1
        #
        #  Catch non-convergence.
        #
    if ka != 1:
        raise Exception('ABWE2 - Fatal error!')

        #
        #  Computation of the weight.
        #
    an = n

    w2 = 2.0 / (an * pd2 * p0)

    p1 = 0.0
    p2 = b[m + 1 - 1]
    yy = 4.0 * x * x - 2.0
    for k in range(1, m + 1):
        i = m - k + 1
        p0 = p1
        p1 = p2
        p2 = yy * p1 - p0 + b[i - 1]

    if even:
        w1 = w2 + coef2 / (pd2 * x * (p2 - p1))
    else:
        w1 = w2 + 2.0 * coef2 / (pd2 * (p2 - p0))

    return x, w1, w2


@nb.njit(fastmath=True)
def kronrod_nodes_and_weight_raw(n: int, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    n : int
        the order of the Gauss rule.
    tol : float
        the requested absolute accuracy of the abscissas.

    Returns
    -------
    res : tuple[np.ndarray, np.ndarray, np.ndarray
        tuple contains  the abscissas, the Gauss-Kronrod weights, and the Gauss weights.

    """
    m = ((n + 1) // 2)
    even: bool = (2 * m == n)

    b = np.zeros(m + 1)
    tau = np.zeros(m)
    w1 = np.zeros(n + 1)
    w2 = np.zeros(n + 1)
    x = np.zeros(n + 1)

    d = 2.0
    an = 0.0
    for k in range(1, n + 1):
        an = an + 1.0
        d = d * an / (an + 0.5)
    #
    #  Calculation of the Chebyshev coefficients of the orthogonal polynomial.
    #
    tau[1 - 1] = (an + 2.0) / (an + an + 3.0)
    b[m - 1] = tau[1 - 1] - 1.0
    ak = an

    for l in range(1, m):
        ak = ak + 2.0
        tau[l + 1 - 1] = ((ak - 1.0) * ak - an * (an + 1.0)) * (ak + 2.0) * tau[l - 1] / (ak * ((ak + 3.0) * (ak + 2.0)
                                                                                                - an * (an + 1.0)))
        b[m - l - 1] = tau[l + 1 - 1]

        for ll in range(1, l + 1):
            b[m - l - 1] = b[m - l - 1] + tau[ll - 1] * b[m - l + ll - 1]

    b[m + 1 - 1] = 1.0
    #
    #  Calculation of approximate values for the abscissas.
    #
    bb = np.sin(0.5 * np.pi / (an + an + 1.0))
    x1 = np.sqrt(1.0 - bb * bb)
    s = 2.0 * bb * x1
    c = np.sqrt(1.0 - s * s)
    coef = 1.0 - (1.0 - 1.0 / an) / (8.0 * an * an)
    xx = coef * x1
    #
    #  Coefficient needed for weights.
    #
    #  COEF2 = 2^(2*n+1) * n! * n! / (2n+1)!
    #
    coef2 = 2.0 / (2 * n + 1)
    for i in range(1, n + 1):
        coef2 = coef2 * 4.0 * i / (n + i)
    #
    #  Calculation of the K-th abscissa (a Kronrod abscissa) and the
    #  corresponding weight.
    #
    for k in range(1, n + 1, 2):

        [xx, w1[k - 1]] = abwe1(n, m, tol, coef2, even, b, xx)
        w2[k - 1] = 0.0

        x[k - 1] = xx
        y = x1
        x1 = y * c - bb * s
        bb = y * s + bb * c

        if k == n:
            xx = 0.0
        else:
            xx = coef * x1
        #
        #  Calculation of the K+1 abscissa (a Gaussian abscissa) and the
        #  corresponding weights.
        #
        [xx, w1[k + 1 - 1], w2[k + 1 - 1]] = abwe2(n, m, tol, coef2, even, b, xx)

        x[k + 1 - 1] = xx
        y = x1
        x1 = y * c - bb * s
        bb = y * s + bb * c
        xx = coef * x1
    #
    #  If N is even, we have one more Kronrod abscissa to compute,
    #  namely the origin.
    #
    if even:
        xx = 0.0
        [xx, w1[n + 1 - 1]] = abwe1(n, m, tol, coef2, even, b, xx)
        w2[n + 1 - 1] = 0.0
        x[n + 1 - 1] = xx

    return x, w1, w2


@nb.njit(fastmath=True)
def kronrod_weight_and_nodes(n: int, tol: float = 1e-5):
    """

    Parameters
    ----------
    n
    tol

    Returns
    -------

    """
    x, w1, w2 = kronrod_nodes_and_weight_raw(n, tol)
    x_gk_arranged = np.concatenate((-x[:-1], np.flip(x)))
    w_gk_arranged = np.concatenate((w1[:-1], np.flip(w1)))
    return x_gk_arranged, w_gk_arranged


# @nb.njit(fastmath=True)
def one_d_kronrod_weight_and_nodes(a_level: int) -> list[OneDNodesAndWeights]:
    """

    Parameters
    ----------
    a_level : int
        a level

    Returns
    -------
    one_d_nodes_and_weight : list
        list of the form (leve, roots, weights)

    References
    ---------
    https://people.math.sc.edu/Burkardt/m_src/sparse_grid_cc/sparse_grid_cc.html
    """

    one_d_nodes_and_weights_list = []
    polynomial_orders = np.power(2, np.arange(1, a_level + 1)) - 1

    for i in range(a_level):
        polynomial_order = polynomial_orders[i]
        nodes, weights = kronrod_weight_and_nodes(polynomial_order)
        one_d_nodes_and_weights_list.append(OneDNodesAndWeights(i + 1, nodes, weights))

    return one_d_nodes_and_weights_list


def sparse_kronrod_quadrature(dim: int, level: int):
    """

    Parameters
    ----------
    dim
    level

    Returns
    -------

    Examples
    --------
    """
    one_d_gh_nodes_and_weights_lists = one_d_kronrod_weight_and_nodes(level)
    smol_indices, grid_points, weights = smolyak_grid(dim, one_d_gh_nodes_and_weights_lists)

    return smol_indices, grid_points, weights
