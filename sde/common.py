from collections import namedtuple
import jax.numpy as jnp
from typing import Tuple, List

SDEpair = namedtuple('SDEpair', ['drift', 'diffusion'])
ButcherTable = namedtuple('ButcherTable', ['name', 'step', 'a', 'b', 'c', 'A', 'B'])


def time(nt: int, interval: Tuple = (0.0, 1.0)) -> Tuple[float, jnp.ndarray]:
    """
    The function splits the time interval into parts
    Parameters
    ----------
    nt       :int
        time parition
    interval: Tuple (optional)
        t_0, t_end default (0.,1.)

    Returns
    -------
    out: Tuple
        dt, t
    """

    """The function splits the time interval into parts"""
    (t_0, t_end) = interval

    dt = (t_end - t_0) / float(nt)
    t = jnp.arange(1, nt + 1, 1) * dt
    return dt, t
