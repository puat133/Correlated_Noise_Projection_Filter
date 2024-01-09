import jax.numpy as jnp
from jax import jit
from functools import partial


def a_hash_function(ar: jnp.ndarray):
    return int(jnp.sum(ar))


class HashableArrayWrapper:

    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return a_hash_function(self.val)

    def __eq__(self, other):
        return isinstance(other, HashableArrayWrapper) and jnp.all(jnp.equal(self.val, other.val))


def array_jit(fun, static_array_argnums=()):
    @partial(jit, static_argnums=static_array_argnums)
    def callee(*args):
        args = list(args)
        for i in static_array_argnums:
            args[i] = args[i].val
        return fun(*args)

    def caller(*args):
        args = list(args)
        for i in static_array_argnums:
            args[i] = HashableArrayWrapper(args[i])
        return callee(*args)

    return caller
