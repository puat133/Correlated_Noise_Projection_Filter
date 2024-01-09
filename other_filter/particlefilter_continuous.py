from functools import partial
from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from other_filter.resampling import normalize_log_weights
from other_filter.particlefilter import ParticleFilter
from other_filter.resampling import log_gaussian_density, log_mean_exp, systematic_or_stratified, multinomial


class ContinuousParticleFilter(ParticleFilter):
    def __init__(self,
                 n_devices: int,
                 n_particle_per_device: int,
                 initial_samples: jnp.ndarray,
                 measurement_history: jnp.ndarray,
                 process_drift: Callable,
                 process_diffusion: Callable,
                 measurement_drift: Callable,
                 measurement_diffusion: Callable,
                 process_brownian_dim: int,
                 dt: float,
                 constraint: Callable,
                 prng_key: jnp.ndarray,
                 process_cov: jnp.ndarray = None,
                 resampling='systematic'):
        r"""



        Parameters
        ----------
        n_devices
        n_particle_per_device
        initial_samples
        measurement_history
        process_drift
        process_diffusion
        measurement_drift
        measurement_diffusion
        process_brownian_dim
        dt
        constraint
        prng_key
        resampling
        """

        super().__init__(n_devices,
                         n_particle_per_device,
                         initial_samples,
                         measurement_history,
                         None,
                         None,
                         constraint,
                         jnp.eye(initial_samples.shape[-1]),  # feeding identity in measurement and process covariances
                         jnp.eye(measurement_history.shape[-1]),
                         prng_key,
                         )

        # in continuous measurement, we need another information
        self._dt = dt
        self._process_brownian_dim = process_brownian_dim
        self._process_drift = process_drift
        self._process_diffusion = process_diffusion
        self._measurement_drift = measurement_drift
        self._measurement_diffusion = measurement_diffusion
        self._dy = jnp.diff(self._measurement_history, axis=0)
        # unfortunately we need to destroy self._measurement_history to keep reuse the original code
        self._measurement_history = self._dy
        self._uniforms = self._uniforms[:-1]
        self._resampling = resampling

        self._process_cov = jnp.eye(self._process_brownian_dim)
        self._mean_zero = jnp.zeros((self._process_brownian_dim,))
        if isinstance(process_cov,jnp.ndarray):
            self._process_cov = process_cov

        def _particle_filter_body_systematic(carry_: tuple, inputs_: tuple):
            x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            dy_, uni_ = inputs_


            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.

            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            dw_ = jnp.sqrt(self._dt) * jrandom.multivariate_normal(subkey_,self._mean_zero,self._process_cov,
                                                                   (self._n_devices,
                                                                    self._n_particle_per_device)
                                                                   )
            x_particle_, log_weights_ = self._parallelized_routine(x_particle_resampled_,
                                                                   dw_,
                                                                   dy_
                                                                   )
            x_particle_resampled_ = systematic_or_stratified(x_particle_, log_weights_, uni_)
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_)

        def _particle_filter_body_stratified(carry_: tuple, inputs_: tuple):
            x_particle_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            dy_, _ = inputs_
            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            # generate an array of uniform random number to be used for a stratified sampling
            uni_ = jrandom.uniform(subkey_, (self._n_devices * self._n_particle_per_device,))

            x_particle_resampled_ = systematic_or_stratified(x_particle_, log_weights_, uni_)
            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.

            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            dw_ = jnp.sqrt(self._dt) * jrandom.normal(subkey_, (self._n_devices,
                                                                self._n_particle_per_device,
                                                                self._process_brownian_dim))
            x_particle_, log_weights_ = self._parallelized_routine(x_particle_resampled_,
                                                                   dw_,
                                                                   dy_
                                                                   )
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_)

        def _particle_filter_body_multinomial(carry_: tuple, inputs_: tuple):
            x_particle_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            dy_, _ = inputs_
            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            # generate an array of uniform random number to be used for a stratified sampling
            uni_ = jrandom.uniform(subkey_, (self._n_devices * self._n_particle_per_device + 1,))

            x_particle_resampled_ = multinomial(x_particle_, log_weights_, uni_)
            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.

            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            dw_ = jnp.sqrt(self._dt) * jrandom.normal(subkey_, (self._n_devices,
                                                                self._n_particle_per_device,
                                                                self._process_brownian_dim))
            x_particle_, log_weights_ = self._parallelized_routine(x_particle_resampled_,
                                                                   dw_,
                                                                   dy_
                                                                   )
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_)

        def _particle_filter_body(carry_: tuple, inputs_: tuple):
            dy_, _ = inputs_

            x_particle_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_

            # No resampling
            x_particle_resampled_ = x_particle_

            x_ = jnp.sum(x_particle_resampled_ * jnp.exp(log_weights_[:, :, jnp.newaxis]), axis=(0, 1))
            # take the mean over here.

            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            dw_ = jnp.sqrt(self._dt) * jrandom.normal(subkey_, (self._n_devices,
                                                                self._n_particle_per_device,
                                                                self._process_brownian_dim))
            x_particle_, log_weights_ = self._parallelized_routine(x_particle_resampled_,
                                                                   dw_,
                                                                   dy_
                                                                   )
            # normalize log weight here
            log_weights_ = normalize_log_weights(log_weights_)
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_)

        if self._resampling.lower() == 'systematic':
            self._particle_filter_body = _particle_filter_body_systematic
        elif self._resampling.lower() == 'stratified':
            self._particle_filter_body = _particle_filter_body_stratified
        elif self._resampling.lower() == 'multinomial':
            self._particle_filter_body = _particle_filter_body_multinomial
        else:
            self._particle_filter_body = _particle_filter_body

    @property
    def resampling(self):
        return self._resampling

    @partial(jit, static_argnums=[0, ])
    @partial(jnp.vectorize, signature='(n),(k)->(n),()', excluded=[0, 3])
    def _parallelized_routine(self, x_particle_, dw, dy):
        # use Euler Maruyama
        x_particle_ += self._process_drift(x_particle_) * self._dt + self._process_diffusion(x_particle_) @ dw
        h = self._measurement_drift(x_particle_)
        ell = self._measurement_diffusion(x_particle_)
        log_weights_ = log_gaussian_density(dy - h * self._dt, self._dt * ell @ ell.T)
        return x_particle_, log_weights_
