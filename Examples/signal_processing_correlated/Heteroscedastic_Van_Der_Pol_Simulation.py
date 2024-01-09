import multiprocessing
import os
import pathlib
import logging

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())

import jax
from jax.config import config

config.update("jax_enable_x64", True)  # this must be enabled, or the particle filter simulation will be wrong

DEBUG_MODE = False

if DEBUG_MODE:
    config.update('jax_disable_jit', True)
    config.update("jax_debug_nans", True)

import jax.numpy as jnp
import time
import jax.random as jrandom
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import sympy as sp
import argparse
from datetime import datetime
from utils.density_manipulations import histogram_history, hellinger_distance_history, expectation_history, \
    expectation_history_not_resampled
from utils.plotter import create_animation_from_density, plot_natural_statistics, plot_hellinger_distance
from utils.hdf_io import save_to_hdf
from utils.boolean_parser import add_boolean_argument
from symbolic.n_d import mix_monomials_up_to_order, hyperbolic_cross_monomials
# import skimage.transform as sk_tranform
from other_filter.resampling import normalize_log_weights, essl


def __run_simulation(use_cuda: bool = True,
                     use_multi_core_cpu: bool = True,
                     kappa_vdp: float = 0.0,
                     var_scale: float = 1.0,
                     mu_vdp: float = 1.5,
                     sigma_v_vdp: float = 1.0,
                     sigma_w_vdp: float = 1.0,
                     dt: float = 5e-4,
                     nt: int = 4000,
                     prngkey: int = 0,
                     max_order_monomials: int = 4,
                     n_particle_per_device: int = 10000,
                     qmc_nodes_number: int = 1000,
                     initial_exponential_density_helper_level: int = 7,
                     bijection_parameter_scale_factor: float = 1.4,
                     resampling: str = "systematic",
                     integrator: str = "spg",
                     projection_filter_sde_solver: str = "euler",
                     projection_filter_spg_rule: str = "gauss-patterson",
                     projection_filter_spg_level: int = 6,
                     grid_limits: jnp.array = jnp.array([[-10., 10.], [-10., 10.]]),  # (x_min,x_max,y_min,y_max)
                     num_point_per_axis: int = 150,
                     weight_cut_off: float = None,
                     compare: bool = False,
                     save: bool = False,
                     animate: bool = False,
                     plot_results: bool = False,
                     gauss_init: bool = False,
                     hyperbolic: bool = False,
                     restricted: bool = True,
                     moment_iterations: int = 1,
                     corr_strength: float = 0,
                     k1: float = 5.0,
                     k2: float = 50
                     ):
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from projection_filter.n_d_exponential_family_projection_filter_gaussian_bijection import \
        MultiDimensionalSStarProjectionFilterWithParametrizedBijection
    from projection_filter.n_d_exponential_family_projection_filter_hermite import \
        MultiDimensionalSStarProjectionFilterHermite
    from projection_filter.n_d_exponential_family_sparse import MultiDimensionalExponentialFamilySparse
    from projection_filter.n_d_exponential_family_hermite import MultiDimensionalExponentialFamilyHermite
    from projection_filter.n_d_exponential_family import MultiDimensionalExponentialFamily
    from projection_filter.n_d_exponential_family_spg import MultiDimensionalExponentialFamilySPG
    from projection_filter.n_d_exponential_family_halton import MultiDimensionalExponentialFamilyQMC
    from projection_filter.n_d_exponential_family_projection_filter_correlated_gaussian_bijection import \
        MultiDimensionalCorrelatedFilterWithParametrizedBijection
    from projection_filter.n_d_exponential_family_projection_filter_correlated_hermite import \
        MultiDimensionalCorrelatedFilterHermite
    from projection_filter.n_d_exponential_family_projection_filter_correlated import \
        MultiDimensionalCorrelatedProjectionFilter
    from projection_filter.n_d_exponential_family_hermite import gauss_hermite_bijection

    import symbolic.one_d
    from sde import SDESolverTypes, sde_solver
    from sde.wiener import multidimensional_wiener_process
    from other_filter.particlefilter_continuous import ContinuousParticleFilter
    from other_filter.particlefilter_continuous_correlated import ContinuousParticleFilterCorrelated
    from numerical_pde_nonlinear_filtering.two_d_nonlinear_filter_via_pde_time_independent import \
        TimeIndependentTwoDNonLinearFilterPDE
    from functools import partial

    # Enable debugging
    # jax.debug_infs()
    # jax.debug_nans()

    classes = [MultiDimensionalSStarProjectionFilterWithParametrizedBijection,
               MultiDimensionalSStarProjectionFilterHermite,
               MultiDimensionalCorrelatedProjectionFilter,
               ContinuousParticleFilter,
               MultiDimensionalExponentialFamilySparse,
               MultiDimensionalExponentialFamilyHermite,
               MultiDimensionalExponentialFamily,
               MultiDimensionalExponentialFamilySPG,
               MultiDimensionalExponentialFamilyQMC,
               TimeIndependentTwoDNonLinearFilterPDE,
               list
               ]

    x, dw, dv = sp.symbols(('x1:3', 'dw1:3', 'dv1:3'))
    t = sp.symbols('t')
    const_1 = k1
    const_2 = k2
    a_fun = sp.sqrt(const_1 + (x[0] ** 2 + x[1] ** 2) / const_2)

    def a_fun_jp(x_):
        return jnp.array([jnp.sqrt(const_1 + (x_[0] ** 2 + x_[1] ** 2) / const_2)])

    def F(x: jnp.array, t: float) -> jnp.array:
        """
        The drift part of dynamic and measurement sdes stacked together.

        Parameters
        ----------
        x : jnp.array
            state
        t : float
            time

        Returns
        -------
        F: jnp.array
            result
        """

        return jnp.array([kappa_vdp * x[0] + x[1],
                          kappa_vdp * x[1] + mu_vdp * (1. - x[0] * x[0]) * x[1] - x[0],
                          x[0]])  # Default

    def G(x: jnp.array, t: float):
        """
        The diffusion part of dynamic and measurement sdes stacked together.

        Parameters
        ----------
        x : jnp.array
            state
        t : float
            time

        Returns
        -------
        G: jnp.array
            result
        """
        return jnp.array([[0., 0.],
                          jnp.concatenate([sigma_w_vdp * a_fun_jp(x), jnp.array([0.])]),
                          jnp.concatenate([jnp.array([0.]), sigma_v_vdp * a_fun_jp(x)])
                          ])

    # For continuous particle filter
    @partial(jnp.vectorize, signature='(n)->(n)')
    def f_(x: jnp.ndarray):
        """
        the drift part of the dynamic.
        Parameters
        ----------
        x : jnp.ndarray
            state

        Returns
        -------
        f : jnp.ndarray
            result
        """
        return jnp.array([kappa_vdp * x[0] + x[1], kappa_vdp * x[1] + mu_vdp * (1. - x[0] * x[0]) * x[1] - x[0]])

    @partial(jnp.vectorize, signature='(n)->(n,m)')
    def g_(x: jnp.ndarray):
        """
        the diffusive part of the dynamic.
        Parameters
        ----------
        x : jnp.ndarray
            state

        Returns
        -------
        g : jnp.ndarray
            result
        """
        return jnp.array([[0.], sigma_w_vdp * a_fun_jp(x)])

    @partial(jnp.vectorize, signature='(n)->(1)')
    def h_(x: jnp.ndarray):
        """
        the drift part of the measurement.
        Parameters
        ----------
        x : jnp.ndarray
            state

        Returns
        -------
        h : jnp.ndarray
            result
        """
        return jnp.array([x[0]])

    @partial(jnp.vectorize, signature='(n)->(1,1)')
    def ell_(x: jnp.ndarray):
        """
        the diffusive part of the measurement.
        Parameters
        ----------
        x : jnp.ndarray
            state

        Returns
        -------
        ell : jnp.ndarray
            result
        """
        return jnp.array([sigma_v_vdp * a_fun_jp(x)])

    f = sp.Matrix([kappa_vdp * x[0] + x[1], kappa_vdp * x[1] + mu_vdp * (1. - x[0] * x[0]) * x[1] - x[0]])
    g = sp.Matrix([[0.], [sigma_w_vdp * a_fun]])
    dynamic_sde = symbolic.one_d.SDE(f, g, t, x, dw)
    measurement_sde = symbolic.one_d.SDE(drifts=sp.Matrix([x[0]]),
                                         diffusions=sp.Matrix([[sigma_v_vdp * a_fun]]), time=t,
                                         variables=x,
                                         brownians=dv)

    tspan = jnp.arange(nt) * dt
    prng_key, subkey = jrandom.split(jrandom.PRNGKey(prngkey))
    cov_mat = jnp.array([[1, corr_strength], [corr_strength, 1]])
    dW = multidimensional_wiener_process((nt, 2), dt, subkey, cov_mat)

    # this is mean and covariance for the initial Gauss density
    init_mean = jnp.array([1., 1.])
    var_core = jnp.diag(jnp.array([1., 1.]))
    var_init = var_scale * var_core  #
    # var_init = var_core
    var_init_inv = jnp.linalg.solve(var_init, jnp.eye(2))
    theta_1 = var_init_inv @ init_mean
    theta_2 = -0.5 * var_init_inv

    X0 = onp.array([init_mean[0], init_mean[1], init_mean[0]])
    X_integrated = sde_solver(F, G, X0, tspan, dW, solver_type=SDESolverTypes.ItoEulerMaruyama)
    measurement_record = X_integrated[:, 2:]

    # for dimension of the space of parameter is Four
    if hyperbolic:
        logger.info('Using Hyperbolic cross')
        natural_statistics_symbolic = hyperbolic_cross_monomials(x, max_order_monomials)
    else:
        natural_statistics_symbolic = mix_monomials_up_to_order(x, max_order_monomials)

    # Gaussian initial state
    gaussian_initial_condition = jnp.zeros(len(natural_statistics_symbolic))
    theta_indices_for_mu = []
    # the mean
    for i in range(init_mean.shape[0]):
        for k in range(len(natural_statistics_symbolic)):
            if natural_statistics_symbolic[k] == x[i]:
                gaussian_initial_condition = gaussian_initial_condition.at[k].set(theta_1[i])
                theta_indices_for_mu.append(k)
                break

    theta_indices_for_mu = jnp.array(theta_indices_for_mu)
    theta_indices_for_Sigma = onp.zeros((init_mean.shape[0], init_mean.shape[0]))
    # the variance
    for i in range(init_mean.shape[0]):
        for j in range(i, init_mean.shape[0]):
            multiplier = 2
            if i == j:
                multiplier = 1
            for k in range(len(natural_statistics_symbolic)):
                if natural_statistics_symbolic[k] == x[i] * x[j]:
                    gaussian_initial_condition = gaussian_initial_condition.at[k].set(multiplier * theta_2[i, j])
                    theta_indices_for_Sigma[i, j] = k
                    theta_indices_for_Sigma[j, i] = k
                    break

    theta_indices_for_Sigma = jnp.array(theta_indices_for_Sigma, dtype=jnp.int32)
    theta_indices_for_bijection_params = (theta_indices_for_mu, theta_indices_for_Sigma)
    _a_scale = 1 / var_scale
    _a_mean = init_mean
    x1, x2 = x
    pol = 2 * (_a_scale * (x1 - _a_mean[0]) ** 2) - (_a_scale * (x1 - _a_mean[0])) ** 4 + 2 * (
            _a_scale * (x2 - _a_mean[1]) ** 2) - (_a_scale * (x2 - _a_mean[1])) ** 4
    non_gaussian_initial_condition = jnp.array([float(pol.as_poly(x).coeff_monomial(natural_statistics_symbolic[i]))
                                                for i in range(len(natural_statistics_symbolic))])

    if gauss_init:
        initial_condition = gaussian_initial_condition
        _mu = init_mean
        _Sigma = var_init
        _Sigma_eigvals, _Sigma_eigvects = jnp.linalg.eigh(_Sigma)
        intial_bijection_parameters_gaussian = (_mu, _Sigma_eigvals, _Sigma_eigvects, bijection_parameter_scale_factor)
        intial_bijection_parameters_hermite = (_mu, _Sigma_eigvals, _Sigma_eigvects, bijection_parameter_scale_factor)
    else:
        initial_condition = non_gaussian_initial_condition
        _mu = init_mean
        _Sigma = 1 * var_init
        _Sigma_eigvals, _Sigma_eigvects = jnp.linalg.eigh(_Sigma)
        intial_bijection_parameters_gaussian = (_mu, _Sigma_eigvals, _Sigma_eigvects, bijection_parameter_scale_factor)
        intial_bijection_parameters_hermite = (_mu, _Sigma_eigvals, _Sigma_eigvects, bijection_parameter_scale_factor)

    logger.info("Preparing projection filter simulation(s)")

    em_pfs = []
    integrators = []
    if compare:
        integrators = ['spg', "hermite", "qmc"]
    else:
        integrators.append(integrator)

    if restricted:
        logger.info("Using restricted version projection filter.")
        for integrator in integrators:
            if integrator.lower() == 'spg':
                em_pf = MultiDimensionalCorrelatedFilterWithParametrizedBijection(dynamic_sde,
                                                                                  measurement_sde,
                                                                                  natural_statistics_symbolic,
                                                                                  constants=None,
                                                                                  initial_condition=initial_condition,
                                                                                  measurement_record=measurement_record,
                                                                                  delta_t=dt,
                                                                                  noise_correlation_matrix=jnp.array(
                                                                                      [[corr_strength]]),
                                                                                  bijection_parameters=(
                                                                                      _mu, _Sigma_eigvals,
                                                                                      _Sigma_eigvects,
                                                                                      bijection_parameter_scale_factor),
                                                                                  level=projection_filter_spg_level,
                                                                                  integrator='spg',
                                                                                  nodes_number=qmc_nodes_number,
                                                                                  epsilon=0,
                                                                                  ode_solver=projection_filter_sde_solver,
                                                                                  sRule=projection_filter_spg_rule,
                                                                                  theta_indices_for_bijection_params=
                                                                                  theta_indices_for_bijection_params,
                                                                                  moment_matching_iterations=
                                                                                  moment_iterations
                                                                                  )
            elif integrator.lower() == 'hermite':
                sRule = "hermite"
                em_pf = MultiDimensionalCorrelatedFilterHermite(dynamic_sde,
                                                                measurement_sde,
                                                                natural_statistics_symbolic,
                                                                constants=None,
                                                                initial_condition=initial_condition,
                                                                measurement_record=measurement_record,
                                                                delta_t=dt,
                                                                noise_correlation_matrix=jnp.array(
                                                                    [[corr_strength]]),
                                                                bijection_parameters=(
                                                                    _mu, _Sigma_eigvals,
                                                                    _Sigma_eigvects,
                                                                    bijection_parameter_scale_factor),
                                                                level=projection_filter_spg_level,
                                                                ode_solver=projection_filter_sde_solver,
                                                                theta_indices_for_bijection_params=
                                                                theta_indices_for_bijection_params,
                                                                weight_cut_off=weight_cut_off,
                                                                moment_matching_iterations=
                                                                moment_iterations
                                                                )
            elif integrator.lower() == 'qmc':
                if compare:
                    qmc_nodes_number = 4 * em_pfs[
                        -1].nodes_number  # equates the number of nodes to four times hermite nodes.
                em_pf = MultiDimensionalCorrelatedFilterWithParametrizedBijection(dynamic_sde,
                                                                                  measurement_sde,
                                                                                  natural_statistics_symbolic,
                                                                                  constants=None,
                                                                                  initial_condition=initial_condition,
                                                                                  measurement_record=measurement_record,
                                                                                  delta_t=dt,
                                                                                  noise_correlation_matrix=jnp.array(
                                                                                      [[corr_strength]]),
                                                                                  bijection_parameters=(
                                                                                      _mu, _Sigma_eigvals,
                                                                                      _Sigma_eigvects,
                                                                                      2 * bijection_parameter_scale_factor),
                                                                                  level=projection_filter_spg_level,
                                                                                  integrator='qmc',
                                                                                  nodes_number=qmc_nodes_number,
                                                                                  epsilon=0,
                                                                                  ode_solver=projection_filter_sde_solver,
                                                                                  sRule=projection_filter_spg_rule,
                                                                                  theta_indices_for_bijection_params=
                                                                                  theta_indices_for_bijection_params,
                                                                                  moment_matching_iterations=
                                                                                  moment_iterations
                                                                                  )
            em_pfs.append(em_pf)
    else:
        logger.info("Using non restricted projection filter.")
        for integrator in integrators:
            if integrator.lower() == 'spg':
                em_pf = MultiDimensionalCorrelatedProjectionFilter(dynamic_sde,
                                                                   measurement_sde,
                                                                   natural_statistics_symbolic,
                                                                   constants=None,
                                                                   initial_condition=initial_condition,
                                                                   measurement_record=measurement_record,
                                                                   delta_t=dt,
                                                                   noise_correlation_matrix=jnp.array(
                                                                       [[corr_strength]]),
                                                                   bijection_parameters=(
                                                                       _mu, _Sigma_eigvals,
                                                                       _Sigma_eigvects,
                                                                       bijection_parameter_scale_factor),
                                                                   level=projection_filter_spg_level,
                                                                   integrator='spg',
                                                                   nodes_number=qmc_nodes_number,
                                                                   epsilon=0,
                                                                   ode_solver=projection_filter_sde_solver,
                                                                   sRule=projection_filter_spg_rule,
                                                                   theta_indices_for_bijection_params=
                                                                   theta_indices_for_bijection_params,
                                                                   moment_matching_iterations=
                                                                   moment_iterations
                                                                   )
            elif integrator.lower() == 'hermite':
                sRule = "hermite"
                em_pf = MultiDimensionalCorrelatedProjectionFilter(dynamic_sde,
                                                                   measurement_sde,
                                                                   natural_statistics_symbolic,
                                                                   constants=None,
                                                                   initial_condition=initial_condition,
                                                                   measurement_record=measurement_record,
                                                                   delta_t=dt,
                                                                   noise_correlation_matrix=jnp.array(
                                                                       [[corr_strength]]),
                                                                   bijection_parameters=(
                                                                       _mu, _Sigma_eigvals,
                                                                       _Sigma_eigvects,
                                                                       bijection_parameter_scale_factor),
                                                                   level=projection_filter_spg_level,
                                                                   integrator='hermite',
                                                                   nodes_number=qmc_nodes_number,
                                                                   epsilon=0,
                                                                   bijection=gauss_hermite_bijection,
                                                                   ode_solver=projection_filter_sde_solver,
                                                                   sRule=sRule,
                                                                   theta_indices_for_bijection_params=
                                                                   theta_indices_for_bijection_params,
                                                                   moment_matching_iterations=
                                                                   moment_iterations
                                                                   )
            elif integrator.lower() == 'qmc':
                if compare:
                    qmc_nodes_number = 4 * em_pfs[
                        -1].nodes_number  # equates the number of nodes to four times hermite nodes.
                em_pf = MultiDimensionalCorrelatedProjectionFilter(dynamic_sde,
                                                                   measurement_sde,
                                                                   natural_statistics_symbolic,
                                                                   constants=None,
                                                                   initial_condition=initial_condition,
                                                                   measurement_record=measurement_record,
                                                                   delta_t=dt,
                                                                   noise_correlation_matrix=jnp.array(
                                                                       [[corr_strength]]),
                                                                   bijection_parameters=(
                                                                       _mu, _Sigma_eigvals,
                                                                       _Sigma_eigvects,
                                                                       2 * bijection_parameter_scale_factor),
                                                                   level=projection_filter_spg_level,
                                                                   integrator='qmc',
                                                                   nodes_number=qmc_nodes_number,
                                                                   epsilon=0,
                                                                   ode_solver=projection_filter_sde_solver,
                                                                   sRule=projection_filter_spg_rule,
                                                                   theta_indices_for_bijection_params=
                                                                   theta_indices_for_bijection_params,
                                                                   moment_matching_iterations=
                                                                   moment_iterations
                                                                   )
            em_pfs.append(em_pf)

    logger.info("Running the projection filter(s)..")
    execution_times = onp.zeros((len(em_pfs),))
    for i, em_pf in enumerate(em_pfs):
        logger.info("Executing #{} projection filter".format(i + 1))
        start_time = time.time()
        em_pf.propagate()
        em_pf.state_history.block_until_ready()
        execution_times[i] = time.time() - start_time
        logger.info(
            "Completed running #{} projection filter. Execution time = {} seconds".format(i + 1, execution_times[i]))

    logger.info("Preparing Particle filter simulation..")
    if use_multi_core_cpu:
        n_devices = jax.local_device_count()
    else:
        n_devices = 1
    prng_key, subkey = jrandom.split(prng_key)

    if onp.abs(corr_strength) > 0:
        cpf = ContinuousParticleFilterCorrelated(n_devices=n_devices,
                                                 n_particle_per_device=n_particle_per_device,
                                                 initial_samples=
                                                 em_pfs[0].exponential_density.sample(
                                                     (n_devices, n_particle_per_device), initial_condition,
                                                     theta_indices_for_mu,
                                                     intial_bijection_parameters_gaussian,
                                                     subkey),
                                                 measurement_history=measurement_record,
                                                 process_drift=f_,
                                                 process_diffusion=g_,
                                                 measurement_drift=h_,
                                                 measurement_diffusion=ell_,
                                                 process_brownian_dim=1,
                                                 dt=dt,
                                                 constraint=None,
                                                 prng_key=prng_key,
                                                 noise_correlation_matrix=jnp.array([[corr_strength]]),
                                                 resampling=resampling
                                                 )
    else:
        cpf = ContinuousParticleFilter(n_devices=n_devices,
                                       n_particle_per_device=n_particle_per_device,
                                       initial_samples=
                                       em_pfs[0].exponential_density.sample(
                                           (n_devices, n_particle_per_device), initial_condition,
                                           theta_indices_for_mu,
                                           intial_bijection_parameters_gaussian,
                                           subkey),
                                       measurement_history=measurement_record,
                                       process_drift=f_,
                                       process_diffusion=g_,
                                       measurement_drift=h_,
                                       measurement_diffusion=ell_,
                                       process_brownian_dim=1,
                                       dt=dt,
                                       constraint=None,
                                       prng_key=prng_key,
                                       resampling=resampling
                                       )
    # if False:
    logger.info("Running particle filter across {} - cpus".format(n_devices))
    start_time = time.time()
    _, _, x_particle_history, neg_likelihood_history, log_weights_history, _ = cpf.run()
    x_particle_history.block_until_ready()
    particle_filter_execution_time = time.time() - start_time
    logger.info(
        "Completed running particle filter filter. Execution time = {} seconds".format(particle_filter_execution_time))

    logger.info("Particle Filter simulation completed. Now reshaping the vector")
    x_particle_history = onp.asarray(x_particle_history).reshape((nt, n_devices * n_particle_per_device, 2))
    log_weights_history = log_weights_history.reshape((nt, n_devices * n_particle_per_device))
    log_weights_history = jnp.vectorize(normalize_log_weights, signature='(n)->(n)')(log_weights_history)
    essl_history = jnp.vectorize(essl, signature='(n)->()')(log_weights_history)

    logger.info("Getting the densities ...")
    num_points = jnp.array([num_point_per_axis, num_point_per_axis], dtype=jnp.int32)
    statistics_str = [str(stat).replace('**', '^').replace(
        'x1', 'x_1').replace('x2', 'x_2').replace('*', '') for stat in em_pf.natural_statistics_symbolic]

    density_histories = []
    bijected_points_histories = []
    for em_pf in em_pfs:
        density_history, bijected_points_history = em_pf.get_density_values(grid_limits, num_points)
        density_histories.append(density_history)
        bijected_points_histories.append(bijected_points_history)

    dxs = jnp.diff(grid_limits, axis=-1).squeeze() / num_points
    xbins = jnp.linspace(grid_limits[0, 0] - dxs[0], grid_limits[0, 1] + dxs[0], num_point_per_axis + 1)
    ybins = jnp.linspace(grid_limits[1, 0] - dxs[1], grid_limits[1, 1] + dxs[1], num_point_per_axis + 1)

    # this is only valid if resampling method is used.
    empirical_den_pf_history = histogram_history(x_particle_history, xbins, ybins)

    logger.info("Densities are calculated completely.")

    logger.info("Calculating Hellinger distances ...")
    hell_dist_hists = []
    srules = []

    ground_truth_density_history = empirical_den_pf_history

    for i in range(len(em_pfs)):
        hell_dist_hist = hellinger_distance_history(density_histories[i], ground_truth_density_history, dxs)
        hell_dist_hists.append(hell_dist_hist)
        srules.append(em_pfs[i].exponential_density.srule)

    logger.info("Calculating moments ...")
    if cpf.resampling.lower() in ['systematic', 'stratified', 'multinomial']:
        moments_particle_filter = expectation_history(em_pfs[0].exponential_density.natural_statistics,
                                                      x_particle_history)
    else:
        moments_particle_filter = expectation_history_not_resampled(em_pfs[0].exponential_density.natural_statistics,
                                                                    log_weights_history,
                                                                    x_particle_history)
    moments_particle_filter = moments_particle_filter
    moments_projection_filters = []
    rel_ent_histories = []
    for em_pf in em_pfs:
        moments_projection_filter = em_pf.get_natural_statistics_expectation()
        moments_projection_filters.append(moments_projection_filter)
        rel_ent_history = em_pf.empirical_kld(x_particle_history)
        rel_ent_histories.append(rel_ent_history)

    rel_ent_histories = jnp.stack(rel_ent_histories)

    if save:
        logger.info("Saving variables to hdf file")
        save_to_hdf('./simulation_result/{}/variables.hdf'.format(simulation_id), locals(),
                    classes=classes,
                    excluded_vars=['x_particle_history',
                                   '_q_particle',
                                   '_uniforms',
                                   'log_weights_history'])

    if plot_results:
        # set some matplotlib variables
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            'text.latex.preamble': r'\usepackage{amsfonts}'
        })
        sns.set_context("poster")

        logger.info("Plotting ESSL of weights ...")
        plt.plot(tspan, essl_history, linewidth=0.5)
        plt.xlabel('t')
        plt.ylabel('ESSL')
        plt.tight_layout()
        plt.savefig('./simulation_result/{}/essl.pdf'.format(simulation_id))
        plt.close()

        logger.info("Plotting empirical KL divergencec ...")
        for i, integrator in enumerate(integrators):
            plt.plot(tspan, rel_ent_histories[i, :], linewidth=0.5, label=integrator)
        plt.xlabel('$t$')
        plt.ylabel('empirical KL divergence')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./simulation_result/{}/rel_ent.pdf'.format(simulation_id))
        plt.close()

        try:
            logger.info("Plotting Hellinger distance ...")
            plot_hellinger_distance(hell_dist_hists=hell_dist_hists,
                                    tspan=tspan,
                                    integrators=integrators,
                                    simulation_id=simulation_id,
                                    pde_included=False)
        except:
            pass

        # try:
        logger.info("Plotting natural statistics ...")
        plot_natural_statistics(moments_projection_filters=moments_projection_filters,
                                moments_particle_filter=moments_particle_filter,
                                statistics_str=statistics_str,
                                tspan=tspan,
                                simulation_id=simulation_id,
                                integrators=integrators  # they are uniform across list member
                                )
        # except:
        #     pass

        if animate:
            logger.info("Creating density animation ...")
            create_animation_from_density(bijected_points_histories=bijected_points_histories,
                                          integrators=integrators,
                                          density_histories=density_histories,
                                          empirical_den_pf_history=empirical_den_pf_history,
                                          grid_limits=grid_limits,
                                          skip_frame=nt // 100,
                                          dt=dt,
                                          simulation_id=simulation_id,
                                          show_nodes=True,
                                          row_col=(2, 2))
    logger.info("Completed. Enjoy ...")


if __name__ == '__main__':
    simulation_time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    simulation_id = "HeteroSe_VDP" + simulation_time_string
    sim_path = pathlib.Path('./simulation_result/{}'.format(simulation_id))
    if not sim_path.exists():
        sim_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Simulation")
    log_file = sim_path / "simulation.log"
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--kappa', default=0.25, type=float, help='kappa for the modified vdp dynamic')
    parser.add_argument('--mu', default=0.3, type=float, help='mu for the modified vdp dynamic')
    parser.add_argument('--sigmaw', default=1.0, type=float, help='scaling factor for the dyanmic Wiener process')
    parser.add_argument('--sigmav', default=1.0, type=float, help='scaling factor for the measurement Wiener process')
    parser.add_argument('--dt', default=2.5e-3, type=float, help='sampling time')
    parser.add_argument('--varscale', default=1, type=float,
                        help='Scalar multiplier to identity for initial variance'
                        )
    parser.add_argument('--k1', default=2.5, type=float, help='k1 constant in diffusion part')
    parser.add_argument('--k2', default=250, type=float,
                        help='k1 constant in diffusion part'
                        )
    parser.add_argument('--corrstrength', default=0.5, type=float,
                        help='correlation strength between dW and dV. By default it is 0.99')
    parser.add_argument('--nt', default=400, type=int, help='time samples')
    parser.add_argument('--qmc_nodes_number', default=2000, type=int, help='nodes_number for quasi monte carlo method')
    parser.add_argument('--seed', default=0, type=int, help='prngkey for jax.random')
    parser.add_argument('--moment_iterations', default=2, type=int, help='moment_matching_iteration_number')
    parser.add_argument('--nparticle', default=1000, type=int, help='number of particle for particle filter for each '
                                                                    'processing device')
    parser.add_argument('--scale', default=1.3, type=float, help='bijection parameter scale factor')
    parser.add_argument('--weightcutoff', default=1e-9, type=float, help='weight cut off for Gauss-Hermite')
    parser.add_argument('--sdesolver', default="euler", type=str, help='sde solver for projection filter')
    parser.add_argument('--resampling', default="systematic", type=str, help='resampling method for particle filter')
    parser.add_argument('--integrator', default="qmc", type=str,
                        help='integrator type for projection filter')
    parser.add_argument('--srule', default="gauss-patterson", type=str, help='sparse grid integration rule')
    parser.add_argument('--slevel', default=4, type=int, help='sparse grid integration level')
    parser.add_argument('--maxgrid', default=8., type=float, help='maximum grid limit')
    parser.add_argument('--npoint', default=200, type=int, help='number of particles per dimension')
    parser.add_argument('--order', default=4, type=int, help='maximum total order of monomials in the natural '
                                                             'statistics')

    add_boolean_argument(parser, 'cuda', default=False, messages="whether to use cuda or not")
    add_boolean_argument(parser, 'compare', default=False, messages="whether to compare Gauss-Hermite and "
                                                                    "Gauss-Patterson at the same time or not.")
    add_boolean_argument(parser, "multicore", default=True,
                         messages="whether to consider each core as separate processing unit")
    add_boolean_argument(parser, "restricted", default=True, messages="whether to use restricted projection filter")
    add_boolean_argument(parser, "save", default=False,
                         messages="whether to save the variables to a hdf file.")
    add_boolean_argument(parser, "plot", default=False,
                         messages="whether to generate some plots or not.")
    add_boolean_argument(parser, "animate", default=False,
                         messages="whether to create density animations or not.")
    add_boolean_argument(parser, "gaussinit", default=True,
                         messages="whether to use Gaussian initial condition or not.")
    add_boolean_argument(parser, 'hyperbolic', default=False, messages="whether to use hyperbolic-cross or not")

    args = parser.parse_args()
    grid_limits = jnp.array([[-args.maxgrid, args.maxgrid], [-args.maxgrid, args.maxgrid]])
    __run_simulation(use_cuda=args.cuda,
                     use_multi_core_cpu=args.multicore,
                     kappa_vdp=args.kappa,
                     var_scale=args.varscale,
                     sigma_w_vdp=args.sigmaw,
                     mu_vdp=args.mu,
                     sigma_v_vdp=args.sigmav,
                     dt=args.dt,
                     nt=args.nt,
                     prngkey=args.seed,
                     max_order_monomials=args.order,
                     n_particle_per_device=args.nparticle,
                     qmc_nodes_number=args.qmc_nodes_number,
                     bijection_parameter_scale_factor=args.scale,
                     resampling=args.resampling,
                     integrator=args.integrator,
                     projection_filter_sde_solver=args.sdesolver,
                     projection_filter_spg_rule=args.srule,
                     projection_filter_spg_level=args.slevel,
                     grid_limits=grid_limits,
                     num_point_per_axis=args.npoint,
                     weight_cut_off=args.weightcutoff,
                     compare=args.compare,
                     save=args.save,
                     plot_results=args.plot,
                     gauss_init=args.gaussinit,
                     animate=args.animate,
                     moment_iterations=args.moment_iterations,
                     corr_strength=args.corrstrength,
                     k1=args.k1,
                     k2=args.k2,
                     hyperbolic=args.hyperbolic,
                     restricted=args.restricted)
