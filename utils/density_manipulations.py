import jax.numpy as jnp
from functools import partial
from typing import Tuple, Callable
from jax import jit
from jax.lax import scan



@partial(jnp.vectorize, signature='(2,2)->(n,n,2)', excluded=(1,))
def create_2d_grid_from_limits(grid_limits: jnp.ndarray, nb_of_points: jnp.ndarray):
    t0 = jnp.linspace(grid_limits[0, 0], grid_limits[0, 1], nb_of_points[0], endpoint=True)
    t1 = jnp.linspace(grid_limits[1, 0], grid_limits[1, 1], nb_of_points[1], endpoint=True)
    grids = jnp.meshgrid(t0, t1, indexing='xy')
    grids = jnp.stack(grids, axis=-1)
    return grids


@partial(jnp.vectorize, signature='(n,n)->()', excluded=(1,))
def integrate_potential(potential: jnp.ndarray, dxs: Tuple) -> float:
    """
    Integrate a potential function on two-dimensional space. The integration is taken using Trapezoidal rule.

    Parameters
    ----------
    potential: jnp.ndarray
        potential to be integrated
    dxs: Tuple
        delta_x for each axis

    Returns
    -------
    integration_result: float
        integration result
    """
    return jnp.trapz(jnp.trapz(potential, dx=dxs[1], axis=1), dx=dxs[0], axis=0)


@partial(jnp.vectorize, signature='(n,n)->(n,n)', excluded=(1,))
def normalize_density(density: jnp.ndarray, dxs: Tuple):
    """
    Normalized a 2d density. The integration used is trapezoidal.

    Parameters
    ----------
    density: jnp.ndarray
        density to be normalized
    dxs: Tuple
        delta_x for each axis

    Returns
    -------
    normalized_density: jnp.ndarray
    """
    return density / integrate_potential(density, dxs)


@jit
@partial(jnp.vectorize, signature='(n)->()', excluded=[1, 2])
def gaussian_kernel_density(point, samples, bandwidth):
    """

    Parameters
    ----------
    point
    samples
    bandwidth

    Returns
    -------

    """
    scaled_points = bandwidth @ (point - samples).T
    gaussian_kernel = jnp.exp(-0.5 * jnp.sum(jnp.square(scaled_points), axis=0)) / jnp.power(2 * jnp.pi,
                                                                                             0.5 * point.shape[0])
    return jnp.sum(gaussian_kernel) / samples.shape[0]


@partial(jnp.vectorize, signature='(n),(n)->(m,m)', excluded=(2, 3))
def histogram(x_particle, y_particle, xbins, ybins):
    """
    get a two-dimensional histogram from a set of x and y positions (particles).

    Parameters
    ----------
    x_particle: jnp.ndarray
    y_particle: jnp.ndarray
    xbins: int
    ybins: int

    Returns
    -------
    empirical_density: jnp.ndarray
        empirical_density

    """
    density, _, _ = jnp.histogram2d(x_particle, y_particle, bins=[xbins, ybins], density=True)
    return density


@partial(jnp.vectorize, signature='(n,n),(n,n)->()', excluded=(2,))
def hellinger_distance(den_1, den_2, dxs) -> float:
    """
    Compute Hellinger distance between two density

    Parameters
    ----------
    den_1: jnp.ndarray
        first density
    den_2 : jnp.ndarray
        second density
    dxs : Tuple
        delta_x for each axis

    Returns
    -------
    hellinger_distance : float
        calculated hellinger distance
    """
    delta_sqrt_dens = jnp.sqrt(jnp.maximum(den_1, 0)) - jnp.sqrt(jnp.maximum(den_2, 0))
    return jnp.sqrt(0.5 * integrate_potential(jnp.square(delta_sqrt_dens), dxs))


def hellinger_distance_history(den_history_1, den_history_2, dxs) -> jnp.ndarray:
    """

    Parameters
    ----------


    den_history_1: jnp.ndarray
        first density history (N_time x N_x x N_y) 2D, or (N_time x N_x) 1D
    den_history_2
        second density history (N_time x N_x x N_y) 2D, or (N_time x N_x) 1D
    dxs :
        grid space on each axis (Nd) or
        grid space on each axis at each time (N_time x Nd)

    Returns
    -------
    hell_dist_hist : jnp.ndarray
        hellinger distance history (N_time)
    """
    if den_history_1.ndim == 3:
        sample_space_dimension = 2
    elif den_history_1.ndim == 2:
        sample_space_dimension = 1
    else:
        raise NotImplementedError("Only one or two dimensional densities are accepted!")

    @jit
    def scanned_fun_1(_carry, _input):
        den_1, den_2 = _input
        hell = hellinger_distance(den_1, den_2, dxs)
        return _carry, hell

    @jit
    def scanned_fun_2(_carry, _input):
        den_1, den_2, dxs_ = _input
        hell = hellinger_distance(den_1, den_2, dxs_)
        return _carry, hell

    @jit
    def scanned_fun_3(_carry, _input):
        den_1, den_2 = _input
        delta_sqrt_dens = jnp.sqrt(jnp.maximum(den_1, 0)) - jnp.sqrt(jnp.maximum(den_2, 0))
        hell =  0.5 * jnp.trapz(jnp.square(delta_sqrt_dens), dx=dxs[0])
        return _carry, hell

    if sample_space_dimension == 2:
        if dxs.ndim == 1:
            _, hell_dist_hist = scan(scanned_fun_1, [], [den_history_1, den_history_2])
        elif dxs.ndim == 2:
            _, hell_dist_hist = scan(scanned_fun_2, [], [den_history_1, den_history_2, dxs])
        else:
            raise Exception("the dxs need to be either one or two dimension!")
    if sample_space_dimension == 1:
        _, hell_dist_hist = scan(scanned_fun_3, [], [den_history_1, den_history_2])

    return hell_dist_hist


@partial(jnp.vectorize, signature='(2,2)->(2),(m),(n)', excluded=(1,))
def dx_and_bins(grid_limit: jnp.ndarray,
                num_points: jnp.ndarray):
    dxs = (grid_limit[:, 1] - grid_limit[:, 0]) / num_points
    x_bins = jnp.linspace(grid_limit[0, 0] - dxs[0], grid_limit[0, 1] + dxs[0], num_points[0] + 1)
    y_bins = jnp.linspace(grid_limit[1, 0] - dxs[1], grid_limit[1, 1] + dxs[1], num_points[1] + 1)
    return dxs, x_bins, y_bins


def histogram_history(x_particle_history: jnp.ndarray,
                      xbins: jnp.ndarray,
                      ybins: jnp.ndarray= None) -> jnp.ndarray:
    """
    Compute histogram history given particle filter samples history.

    Use this instead of the parallelized version of histogram_2d to avoid out of memory error.

    Parameters
    ----------
    x_particle_history: jnp.ndarray
        particle filter records (N_time x N_samples x N_state)
    xbins: jnp.ndarray
        binning on x axis
    ybins: jnp.ndarray
        binning on y axis

    Returns
    -------
    empirical_den_pf_history : jnp.ndarray
        empirical density (N_time x N_x x N_y) if two D or (N_time x N_x) if one D.
    """

    sample_space_dim = x_particle_history.shape[-1]

    @jit
    def scanned_fun_1(_carry, _input):
        x_particle = _input[:, 0]
        y_particle = _input[:, 1]
        a_density = histogram(x_particle, y_particle, xbins, ybins)
        return _carry, a_density.T  # this need to be transposed to match the convention

    @jit
    def scanned_fun_2(_carry, _input):
        a_particle, x_bin_, y_bin_ = _input
        x_particle = a_particle[:, 0]
        y_particle = a_particle[:, 1]
        a_density = histogram(x_particle, y_particle, x_bin_, y_bin_)
        return _carry, a_density.T  # this need to be transposed to match the convention

    @jit
    def scanned_fun_3(_carry, _input):
        a_particle = _input
        a_density, _ = jnp.histogram(a_particle, bins=xbins, density=True)
        return _carry, a_density

    if  sample_space_dim == 2:
        if xbins.ndim == 1:
            _, empirical_den_pf_history = scan(scanned_fun_1, [], x_particle_history)
        elif xbins.ndim == 2:
            _, empirical_den_pf_history = scan(scanned_fun_2, [], (x_particle_history, xbins, ybins))
        else:
            raise Exception("the bins need to be either one or two dimension!")
    elif sample_space_dim == 1:
        if xbins.ndim == 1:
            _, empirical_den_pf_history = scan(scanned_fun_3, [], x_particle_history)
        else:
            raise Exception("the bins need to in one dimension!")

    return empirical_den_pf_history


def expectation_history(statistics: Callable,
                        x_particle_history: jnp.ndarray) -> jnp.ndarray:
    """

    Parameters
    ----------
    statistics : Callable
        function: n -> m

    x_particle_history: jnp.ndarray
        particle filter records (N_time x N_samples x N_state)

    Returns
    -------

    """

    @jit
    def scanned_fun(_carry, _input):
        samples_at_time_t = _input
        expectation_at_time_t = jnp.mean(statistics(samples_at_time_t), axis=0)
        return _carry, expectation_at_time_t

    _, expect_hist = scan(scanned_fun, [], x_particle_history)
    return expect_hist


def expectation_history_not_resampled(statistics: Callable,
                                      normalized_log_weights: jnp.ndarray,
                                      x_particle_history: jnp.ndarray) -> jnp.ndarray:
    """

    Parameters
    ----------
    statistics : Callable
        function: n -> m
    normalized_log_weights: jnp.ndarray
        normalized log weights (N_time x N_samples)
    x_particle_history: jnp.ndarray
        particle filter records (N_time x N_samples x N_state)

    Returns
    -------

    """

    @jit
    def scanned_fun(_carry, _input):
        samples_at_t, normalized_log_weight_at_t = _input
        expectation_at_t = jnp.sum(statistics(samples_at_t) * jnp.exp(normalized_log_weight_at_t[:, jnp.newaxis]),
                                   axis=0)
        return _carry, expectation_at_t

    _, expect_hist = scan(scanned_fun, [], (x_particle_history, normalized_log_weights))
    return expect_hist
