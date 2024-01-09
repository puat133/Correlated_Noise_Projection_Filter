import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def maxnorm(a: np.ndarray) -> float:
    """
    Calculate maxnorm

    Parameters
    ----------
    a : np.ndarray

    Returns
    -------
    maxnorm : float

    Examples
    --------
    >>> a = np.array([1.,2.,3.])
    >>> maxnorm(a)
    3.0
    """
    return np.max(np.abs(a))


@nb.njit(fastmath=True)
def isclose(a: np.array, b: np.array, rtol: float):
    """

    Check whether two array of the same size is close each other.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : float

    Returns
    -------
    decision : bool

    Examples
    --------
    >>> a = np.array([1,2,3])
    >>> b = np.array([1,2,3+1e-9])
    >>> isclose(a,b,1e-5)
    True

    """
    norm_diff = maxnorm(a - b)
    max_ = max(maxnorm(a), maxnorm(b))
    if max_ > 0:
        result = norm_diff / max_ < rtol
    else:
        result = True
    return result


@nb.vectorize([nb.boolean(nb.float64, nb.float64, nb.float64),
               nb.boolean(nb.float32, nb.float32, nb.float32)])
def isclose_scalar(a: float, b: float, rtol: float) -> bool:
    return np.absolute(a - b) < rtol


@nb.njit(fastmath=True)
def isin_nd(a_array: np.ndarray, b_array: np.ndarray, rtol: float = 1e-8):
    """

    This works like `np.isin` but in multidimensional settings.

    Parameters
    ----------
    a_array
    b_array
    rtol

    Returns
    -------

    Examples
    --------
    >>> a = np.array([1.,0.])
    >>> A = np.array([[0.,0.],[1.,0.]])
    >>> isin_nd(A,a)
    array([False,  True])
    >>> B = np.array([[0., 1.], [0.,0.], [1., 3.]])
    >>> isin_nd(B, A)
    array([False,  True, False])

    """
    result = np.zeros((a_array.shape[0],), dtype=nb.boolean)
    dim = b_array.ndim

    if dim > 1:
        for i in range(a_array.shape[0]):
            a = a_array[i]
            for j in range(b_array.shape[0]):
                if isclose(a, b_array[j], rtol=rtol):
                    result[i] = True
                break

    else:
        for i in range(a_array.shape[0]):
            a = a_array[i]
            if isclose(a, b_array, rtol=rtol):
                result[i] = True
                break
    return result