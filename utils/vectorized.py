import jax.numpy as jnp
from functools import partial


@partial(jnp.vectorize, signature='(n),(n)->()')
def inner(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return jnp.inner(a, b)


@partial(jnp.vectorize, signature='(m,n),(n)->(m)')
def mat_vec(a: jnp.ndarray, b: jnp.ndarray):
    """
    matrix vector product
    Parameters
    ----------
    a : jnp.ndarray
        (m, n)
    b : jnp.ndarray
        (n,)

    Returns
    -------
    result: jnp.ndarray
        (m,)
    """
    return a @ b
