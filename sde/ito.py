from functools import partial

import jax.numpy as jnp
import jax.random as jrandom
# from jax.ops import index, index_update

from sde.wiener import multidimensional_wiener_process, scalar_wiener_process


@partial(jnp.vectorize, signature='(n),(n)->(n,n)')
def outer(avec: jnp.ndarray, bvec: jnp.ndarray) -> jnp.ndarray:
    """
    vectorized version of numpy outer product

    Parameters
    ----------
    avec
    bvec

    Returns
    -------

    """
    return jnp.outer(avec, bvec)


# ------------------------------------------------- ----------------------------
# Functions for generating multiple Ito integrals
# ------------------------------------------------- ----------------------------
def ito_1_w1(dw):
    """
    Single Ito integral from 1 to dW for scalar Wiener process
    
    Parameters
    ----------
    dw      :np.ndarray
        brownian increment

    Returns
    -------
    output  :np.ndarray
        dw
    """

    return dw


def ito_2_w1(dw: jnp.ndarray, dt: float, prngkey: jnp.ndarray):
    """
    Generation of double integral values Ito for the scalar Wiener process

    Parameters
    ----------
    dw      : np.ndarray
        brownian increment
    dt      : float
        delta t
    prngkey :np.ndarray
        jax random key

    Returns
    -------

    """

    dzeta = scalar_wiener_process(dw.shape[0], dt, prngkey) + dt

    ito_1_1 = 0.5 * (dw ** 2 - dt)
    ito_1_0 = 0.5 * dt * (dw + dzeta / jnp.sqrt(3))
    ito_0_1 = 0.5 * dt * (dw - dzeta / jnp.sqrt(3))
    ito_0_0 = dt * jnp.ones(dw.shape[0])
    ito = jnp.array([[ito_0_0, ito_0_1], [ito_1_0, ito_1_1]])
    return jnp.swapaxes(ito, 0, 2)


def levi_integral(dw: jnp.ndarray, dt: float, prngkey: jnp.ndarray):
    """
    The Lévy integral for calculating the approximation of the Ito integral
    num --- the number of members of the series

    Parameters
    ----------
    prngkey : np.ndarray
        jax random key
    dw      : ndarray
        Brwonian increment with array (nt, dim)
    dt      : float
        delta time

    Returns
    -------
    out     : np.ndarray
        LeviIntegration array with shape (nt, dim, dim)
    """
    (nt, dim) = dw.shape
    key, subkey = jrandom.split(prngkey)
    x = jrandom.normal(subkey, (nt, nt, dim))

    _, subkey = jrandom.split(key)
    y = jrandom.normal(subkey, (nt, nt, dim))

    y_tilde = (y + jnp.sqrt(2 / dt) * dw[jnp.newaxis, :, :])
    a1 = outer(x, y_tilde)
    a2 = outer(y_tilde, x)
    t = jnp.arange(nt)
    factor = (1.0 / (t + 1))
    scaled_difference = (a1 - a2) * factor[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    a = (dt / jnp.pi) * jnp.sum(scaled_difference, axis=0)

    return a


def ito_1_wm(dw, dt):
    """
    Generation of single Ito integral values for multidimensional
    Wiener process

    Parameters
    ----------
    dw      : ndarray
        Brwonian increment with shape (nt, dim)
    dt      : float
        delta time

    Returns
    -------
    output  : np.ndarray
        ito integration with shape (nt, dim+ 1)
    """
    ito = jnp.hstack([jnp.ones((dw.shape[0], 1)) * dt, dw])
    return ito


def ito_2_wm(dw: jnp.ndarray, dt: float, prngkey: jnp.ndarray):
    """
    Calculation of the Ito integral(by exact formula or by approximation)
    Returns
    Parameters
    ----------
    dw          : np.ndarray
        multi dimensional brownian increment, with shape (nt, dim)
    dt          : float
        delta t
    prngkey    : np.ndarray
        jax random key

    Returns
    -------
    out : np.ndarray
        a list of matrices I(i, j) of shape (nt, dim, dim)
    """

    (nt, dim) = dw.shape
    e = jnp.identity(dim)

    key, subkey = jrandom.split(prngkey)
    dzeta = multidimensional_wiener_process((nt, dim), dt, subkey)  # np.random.normal(loc=0, scale=dt, size=(n, m))

    _, subkey = jrandom.split(subkey)
    ito_0_0 = dt * jnp.ones(nt)
    ito_0_1 = 0.5 * dt * (dw - dzeta / jnp.sqrt(3))
    ito_1_0 = 0.5 * dt * (dw + dzeta / jnp.sqrt(3))
    ito_1_1 = outer(dw, dw) - dt * e + levi_integral(dw, dt, subkey)

    ito_0 = jnp.block([[ito_0_0[:, jnp.newaxis, jnp.newaxis], ito_0_1[:, jnp.newaxis, :]]])
    ito_1 = jnp.block([[ito_1_0[:, :, jnp.newaxis], ito_1_1]])
    return jnp.block([[ito_0], [ito_1]])


def ito_3_w1(dw: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Generation of the triple Ito integral
    for the scalar Wiener process

    Parameters
    ----------
    dw      : ndarray
        Brwonian increment
    dt      : float
        delta time

    Returns
    -------
    output  : np.ndarray
        ito integration
    """

    ito_integration = 0.5 * (dw ** 2 - dt)
    ito_integration = (1.0 / 6.0) * dt * (ito_integration ** 3 - 3.0 * dt * dw)
    return ito_integration
