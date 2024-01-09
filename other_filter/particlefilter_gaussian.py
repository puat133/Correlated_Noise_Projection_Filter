from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jrandom

from other_filter.particlefilter import ParticleFilter


class ParticleFilterWithGaussianInitialDensity(ParticleFilter):
    def __init__(self,
                 n_devices: int,
                 n_particle_per_device: int,
                 mean_init: jnp.ndarray,
                 cov_init: jnp.ndarray,
                 measurement_history: jnp.ndarray,
                 transition_fun: Callable,
                 output_fun: Callable,
                 constraint: Callable,
                 process_cov: jnp.ndarray,
                 meas_cov: jnp.ndarray,
                 prng_key: jnp.ndarray):
        """
        Particle filter implementation with systematic resampling accross multiple-core via `jax.pmap`.
        Both initial, process, measurement noises  densities are assumed to be Gaussians.
        If the constraint is specified, then the
        constraint will be evaluated so that all weights that resulted from the calculation of the likelihood function
        :math: `p(y_k| x[k|k])` will be assigned value 0 if the :math: `x[k|k]` lies outside the constraint [1].

        Parameters
        ----------

        n_devices   : int
            how many device. Should not be more than the length of `jax.devices()`
        n_particle_per_device: int
            how many particle to be per device
        mean_init  : (N_state) np.ndarray
            initial mean
        cov_init  : (N_state,N_state) np.ndarray
            initial covariance
        input_history   : (N_time x N_input) np.ndarray
            input record
        measurement_history : (N_time x N_measurement) np.ndarray
            measurement record
        transition_fun  : Callable
            transition function
        output_fun  : Callable
            output function
        constraint  : Callable
            constraint evaluation on the particles state. Should receive (N_state) array and outputs boolean.
        dynamic_parameters  : Namedtuple
            parameters for dynamics
        process_cov : (N_state x N_state) np.ndarray
            process error covariance (assumed to be diagonal)
        meas_cov    : (N_measurement x N_measurement) np.ndarray
            measurement error covariance
        prng_key    :  np.ndarray
            jax.random.PRNGkey
        Returns
        -------
        result : tuple
            neg_log_likelihood_end, state_end, x_particle_history, neg_likelihood_history, log_weights_history, \
               estimated_state_history
        """

        # generating initial samples
        prng_key, subkey = jrandom.split(prng_key)
        a_shape = (n_devices, n_particle_per_device)
        self._mean_init = mean_init
        self._cov_init = cov_init
        initial_samples = jrandom.multivariate_normal(subkey, self._mean_init, self._cov_init, a_shape)

        super().__init__(
            n_devices=n_devices,
            n_particle_per_device=n_particle_per_device,
            initial_samples=initial_samples,
            measurement_history=measurement_history,
            transition_fun=transition_fun,
            output_fun=output_fun,
            constraint=constraint,
            process_cov=process_cov,
            meas_cov=meas_cov,
            prng_key=prng_key
        )
