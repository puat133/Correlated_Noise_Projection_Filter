from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jrandom

from other_filter.particlefilter_continuous import ContinuousParticleFilter


class ContinuousParticleFilterWithGaussianInitialDensity(ContinuousParticleFilter):
    def __init__(self,
                 n_devices: int,
                 n_particle_per_device: int,
                 mean_init: jnp.ndarray,
                 cov_init: jnp.ndarray,
                 measurement_history: jnp.ndarray,
                 process_drift: Callable,
                 process_diffusion: Callable,
                 measurement_drift: Callable,
                 measurement_diffusion: Callable,
                 process_brownian_dim: int,
                 dt: float,
                 constraint: Callable,
                 prng_key: jnp.ndarray,
                 resampling='systematic'):
        r"""
        Particle filter implementation with systematic resampling accross multiple-core via `jax.pmap`.
        Both initial, process and measurement noises densities are assumed to be Gaussians.
        The dynamics of the process and measurements
        are assumed to be given as the following stochastic differential equations:

        .. math:: dx = f(x,t)dt + g(x,t)dW
        .. math:: dy = h(x,t)dt + \ell(x,t)dV

        where both :math:`dW` and :math:`dV` are independent standard Brownian motions.
        It is assumed that :math:`\ell\ell^\top` is invertible for all :math:`x,t`,

        If the constraint is specified, then the
        constraint will be evaluated so that all weights that resulted from the calculation of the likelihood function
        :math: `p(y_k| x[k|k])` will be assigned value 0 if the :math: `x[k|k]` lies outside the constraint [1].


        Parameters
        ----------
        n_devices
        n_particle_per_device
        mean_init
        cov_init
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

        # generating initial samples
        prng_key, subkey = jrandom.split(prng_key)
        a_shape = ()
        self._mean_init = mean_init
        self._cov_init = cov_init
        initial_samples = self._mean_init + jnp.einsum('ij,klj->kli',
                                                       jnp.linalg.cholesky(self._cov_init),
                                                       jrandom.normal(subkey, (n_devices, n_particle_per_device,
                                                                               self._mean_init.shape[0])))

        super().__init__(
            n_devices=n_devices,
            n_particle_per_device=n_particle_per_device,
            initial_samples=initial_samples,
            measurement_history=measurement_history,
            process_drift=process_drift,
            process_diffusion=process_diffusion,
            measurement_drift=measurement_drift,
            measurement_diffusion=measurement_diffusion,
            process_brownian_dim=process_brownian_dim,
            dt=dt,
            constraint=constraint,
            prng_key=prng_key,
            resampling=resampling
        )
