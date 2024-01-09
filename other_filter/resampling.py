import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from typing import Union


@partial(jnp.vectorize, signature='(n),(n,n)->()')
def log_gaussian_density(x: jnp.ndarray, cov: jnp.ndarray) -> float:
    """
    Evaluate log of gaussian density with zero mean and an inverse covariance matrix `cov_inv`.
    Parameters
    ----------
    x       : (N,) np.ndarray
        vector
    cov: (N,N) np.ndarray
        inverse covariance
    Returns
    -------
    out: float
        log gaussian density
    """
    return -0.5 * (x @ jnp.linalg.solve(cov, x) + cov.shape[0] * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(cov)[1])


@partial(jnp.vectorize, signature='(n),(n,n)->()')
def log_gaussian_density_from_cov_inv(x: jnp.ndarray, cov_inv: jnp.ndarray) -> float:
    """
    Evaluate log of gaussian density with zero mean and an inverse covariance matrix `cov_inv`.
    Parameters
    ----------
    x       : (N,) np.ndarray
        vector
    cov_inv : (N,N) np.ndarray
        inverse covariance
    Returns
    -------
    out: float
        log gaussian density
    """
    return -0.5 * ((x @ cov_inv @ x) + cov_inv.shape[0] * jnp.log(2 * jnp.pi) - jnp.linalg.slogdet(cov_inv)[1])


@partial(jnp.vectorize, signature='(n),(n)->(n,n)')
def outer(x: jnp.ndarray, y: jnp.ndarray):
    return jnp.outer(x, y)


@jit
def log_sum_exp(log_weights: jnp.ndarray) -> float:
    """
    Evaluate log of sum of exponential of a vector of log_weights.

    Parameters
    ----------
    log_weights: (N,) np.ndarray
        an array.

    Returns
    -------
    res : float
        result
    """
    max_log_weights = jnp.max(log_weights)
    return max_log_weights + jnp.log(jnp.sum(jnp.exp(log_weights - max_log_weights)))


@jit
def log_mean_exp(log_weights: jnp.ndarray) -> float:
    """
    Evaluate log of mean of exponential of a vector of log_weights.

    Parameters
    ----------
    log_weights: (N,) np.ndarray
        an array.

    Returns
    -------
    res : float
        result
    """
    max_log_weights = jnp.max(log_weights)
    return max_log_weights + jnp.log(jnp.mean(jnp.exp(log_weights - max_log_weights)))


@jit
def essl(log_weigths: jnp.ndarray):
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    log_weigths: (N,) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    w = jnp.exp(log_weigths - log_weigths.max())
    return (w.sum()) ** 2 / jnp.sum(w ** 2)


@jit
def uniform_spacings(uni: jnp.ndarray):
    """ Generate ordered uniform variates in O(N) time.

        Parameters
        ----------
        uni: jnp.ndarray (>0)
            uniform random array (N+1)

        Returns
        -------
        (N,) float ndarray
            the N ordered variates (ascending order)

        Note
        ----
        This is equivalent to::

            from numpy import random
            u = sort(random.rand(N))

        but the line above has complexity O(N*log(N)), whereas the algorithm
        used here has complexity O(N).

        """
    z = jnp.cumsum(-jnp.log(uni))
    return z[:-1] / z[-1]


@jit
def normalize_log_weights(log_weights: jnp.ndarray):
    return log_weights - log_sum_exp(log_weights)


@jit
def inverse_cdf(su, w):
    """Inverse CDF algorithm for a finite distribution.

        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        w: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)

        Returns
        -------
        flat_indices: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """

    def body_scan(carry, inputs):
        def body_while(val):
            j_, s_ = val
            j_ += 1
            s_ += w[j_]
            return j_, s_

        n, j, s = carry
        j, s = lax.while_loop(lambda val: su[n] > val[1], body_while, (j, s))
        n += 1
        return (n, j, s), j

    #   carry contains n, j, and s
    #   initially, carry is (0, 0, w[0])
    _, flat_indices = lax.scan(body_scan, (0, 0, w[0]), jnp.ones(su.shape[0]))
    return flat_indices


@jit
def resample(indices: tuple[jnp.ndarray, jnp.ndarray], particles: jnp.ndarray) -> jnp.ndarray:
    """ A util function to apply indices to state

    Parameters
    ----------
    indices: tuple of two (N_devices x N_samples_per_devices) array
        The indexing of state
    particles: (N_devices x N_samples_per_devices x N_state) array
        Particles


    Returns
    -------
    out: particles
        Rasampled particles

    """
    dev_indices, sample_indices = indices
    n_devices = particles.shape[0]
    n_samples_per_devices = particles.shape[1]

    return particles[dev_indices, sample_indices, :].reshape((n_devices, n_samples_per_devices, -1))


@jit
def _prepare_weights(log_weights: jnp.ndarray) -> tuple[jnp.ndarray, int, int]:
    """

    Parameters
    ----------
    log_weights : (N_devices x N_samples_per_devices ) array,
        log_weights

    Returns
    -------

    """
    n_devices = log_weights.shape[0]
    n_samples_perdevice = log_weights.shape[1]
    n_particles = n_devices * n_samples_perdevice
    normalized_log_weights = normalize_log_weights(log_weights)
    return normalized_log_weights, n_particles, n_samples_perdevice


@jit
def _resample(particles: jnp.ndarray, flat_indices: jnp.ndarray,
              n_samples_per_device: int):
    """

    Parameters
    ----------
    particles: (N_devices x N_samples_per_devices x N_state) array
        Particles
    flat_indices:
        flat index from 0 up to n_particles-1
    n_samples_per_device: int


    Returns
    -------

    """

    # convert back to the indexing on the original array, where the shape is (n_device,n_sample_perdevice).
    indices = (flat_indices // n_samples_per_device, flat_indices % n_samples_per_device)
    resampled_particles = resample(indices, particles)
    return resampled_particles


@jit
def systematic_or_stratified(particles: jnp.ndarray, log_weights: jnp.ndarray,
                             uniform: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """ Applies systematic or stratified resampling to the state

    Parameters
    ----------
    particles: (N_devices x N_samples_per_devices x N_state) array
        Particles
    log_weights: (N_devices x N_samples_per_devices) array
        The log weights
    uniform: float or (N_particle) array
        if it is a systematic sampling then uniform is a sample from uniform distribution
        if it is a stratified sampling then uniform is an array of (N_particle) shape.

    Returns
    -------

        Resampled particles
    """

    log_weights, n_particles, n_samples_per_device = _prepare_weights(log_weights)
    linspace = jnp.arange(log_weights.size, dtype=log_weights.dtype)  # this is linspace
    # obtain indices using inverse_cdf, this will return indices with value from 0 to n_particles-1
    flat_indices = inverse_cdf((uniform + linspace) / n_particles, jnp.exp(log_weights.ravel()))
    resampled_particles = _resample(particles, flat_indices, n_samples_per_device)

    return resampled_particles


@jit
def multinomial(particles: jnp.ndarray, log_weights: jnp.ndarray, uniform: float) -> jnp.ndarray:
    """

    Parameters
    ----------
    particles
    log_weights
    uniform

    Returns
    -------

    """
    log_weights, n_particles, n_samples_per_device = _prepare_weights(log_weights)
    flat_indices = inverse_cdf(uniform_spacings(uniform), jnp.exp(log_weights.ravel()))
    resampled_particles = _resample(particles, flat_indices, n_samples_per_device)
    return resampled_particles


def resample_particle_history(particle_history: jnp.ndarray,
                              log_weight_history: jnp.ndarray,
                              uniform_history: jnp.ndarray,
                              resampling_method: str = 'systematic_or_stratified') \
        -> jnp.ndarray:
    """

    Parameters
    ----------
    particle_history
    log_weight_history
    uniform_history
    resampling_method

    Returns
    -------

    """
    if resampling_method.lower() in ['systematic', 'stratified', 'systematic_or_stratified']:
        resample_fun = systematic_or_stratified
    elif resampling_method.lower() == 'multinomial':
        resample_fun = multinomial
    else:
        raise ValueError("Resampling method not exists")

    @jit
    def _inner(carry, input):
        particles, log_weights, uniform = input
        resampled_particle = resample_fun(particles, log_weights, uniform)
        return None, resampled_particle

    _, resampled_particle_history = lax.scan(_inner,
                                             None,
                                             (particle_history, log_weight_history,
                                              uniform_history))
    return resampled_particle_history
