from functools import partial
from collections.abc import Callable

import jax.numpy as jnp
from jax import jit

from other_filter.particlefilter_continuous import ContinuousParticleFilter
from other_filter.resampling import log_gaussian_density


class ContinuousParticleFilterCorrelated(ContinuousParticleFilter):
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
                 noise_correlation_matrix: jnp.ndarray,
                 process_cov: jnp.ndarray = None,
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
        noise_correlation_matrix
        resampling
        """

        self._S = noise_correlation_matrix
        self._I_nw = jnp.eye(process_brownian_dim)

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
            process_cov=process_cov,
            resampling=resampling
        )

        # this is required if type I dependency is used. But it seems that dependency type II is the correct one.
        # for correlated particle filter we need dy_t and dy_t+1
        # hence we need to modify a bit
        dy = self._measurement_history.copy()
        dy_plus = jnp.concatenate([dy[1:], dy[jnp.newaxis, -1]])
        self._measurement_history = (dy, dy_plus)

    @partial(jit, static_argnums=[0, ])
    @partial(jnp.vectorize, signature='(n),(k)->(n),()', excluded=[0, 3])
    def _parallelized_routine(self, x_particle_, dw, dy):
        # this part due to noise correlation
        # Based on: Particle Filtering With Dependent Noise Processes, IEEE Trans. Sig Proc 2012
        # for type I dependency
        # Eqs no 17, 23,24,25 in Corollary 2

        dy_t, dy_plus = dy
        ell = self._measurement_diffusion(x_particle_)
        h = self._measurement_drift(x_particle_)
        r = ell @ ell.T
        r_inv_ell_s_t = (jnp.linalg.solve(r, ell @ self._S.T))
        s_ell_t_r_inv_ell_s_t = self._S @ ell.T @ r_inv_ell_s_t
        z = self._I_nw - s_ell_t_r_inv_ell_s_t
        chol_z = jnp.linalg.cholesky(z)
        dv = dy_t - (h * self._dt)  # the dy in this equation should be dy from previous time step
        dw_ = r_inv_ell_s_t.T @ dv + chol_z @ dw

        # use Euler Maruyama
        dx = self._process_drift(x_particle_) * self._dt + self._process_diffusion(x_particle_) @ dw_
        x_particle_ += dx
        h = self._measurement_drift(x_particle_)
        ell = self._measurement_diffusion(x_particle_)
        log_weights_ = log_gaussian_density(dy_plus - h * self._dt, self._dt * ell @ ell.T)
        return x_particle_, log_weights_

    @partial(jit, static_argnums=[0, ])
    @partial(jnp.vectorize, signature='(n),(k)->(n),()', excluded=[0, 3])
    def _parallelized_routine_type_2(self, x_particle_, dw, dy):
        # this part due to noise correlation
        # Based on: Particle Filtering With Dependent Noise Processes, IEEE Trans. Sig Proc 2012
        # for type II dependency
        # Eqs no 35 in Corollary 2

        g_x_min = self._process_diffusion(x_particle_)
        f_x_min = self._process_drift(x_particle_)

        # use Euler Maruyama
        dx = f_x_min * self._dt + g_x_min @ dw
        x_particle_ += dx

        sol, _, _, _ = jnp.linalg.lstsq(g_x_min, dx - f_x_min * self._dt)
        correction_term = self._S.T @ sol
        h = self._measurement_drift(x_particle_)
        ell = self._measurement_diffusion(x_particle_)
        log_weights_ = log_gaussian_density(dy - (h * self._dt + correction_term),
                                            self._dt * (ell @ ell.T - self._S.T @ self._S))
        return x_particle_, log_weights_
