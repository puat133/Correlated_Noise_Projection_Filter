from abc import ABC, abstractmethod
from collections.abc import Callable
import jax.random as jrandom
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

import other_filter.resampling as resampling


class ExponentialFamily(ABC):
    """
    This class encapsulate the exponential family concept, where the log_paritition of this family is computed
    via Numerical Integration and bijection.
    """

    def __init__(self,
                 sample_space_dimension: int,
                 bijection: Callable[[jnp.ndarray, tuple], jnp.ndarray],
                 statistics: Callable[[jnp.ndarray], jnp.ndarray],
                 remaining_statistics: [[jnp.ndarray], jnp.ndarray] = None,
                 bijection_parameters: tuple = None):
        bijection_vectorization_signature: str = '(d)->(d)'
        statistics_vectorization_signature: str = '(d)->(m)'
        partition_vectorization_signature: str = '(m)->()'

        self._sample_space_dim = sample_space_dimension
        self._bijection_params = bijection_parameters
        self._bijection = jnp.vectorize(bijection, signature=bijection_vectorization_signature, excluded=(1,))
        # if sample_space_dimension == 1:
        #     dvolume = jnp.vectorize(jax.grad(bijection), signature=bijection_vectorization_signature,
        #                             excluded=(1,))
        #
        #     self._log_dvolume = lambda _x, _params: jnp.log(dvolume(_x, _params))
        #
        # else:

        # this returns the diagonal elements of jacobian. There should be a better solution to this problem
        self._bijection_jac = jit(jnp.vectorize(jax.jacobian(self._bijection), signature='(d)->(d,d)',
                                                excluded=(1,)))

        # dvolume is the absolute value of the jacobian determinant
        def log_dvolume(_x, _params):
            jac = self._bijection_jac(_x, _params)
            _, log_det = jnp.linalg.slogdet(jac)
            return log_det

        self._log_dvolume = jit(jnp.vectorize(log_dvolume, signature='(d)->()', excluded=(1,)))

        self._stats_vect_sign = statistics_vectorization_signature
        self._par_vect_sign = partition_vectorization_signature

        self._natural_statistics = jit(jnp.vectorize(statistics, signature=self._stats_vect_sign))

        # if sample_space_dimension == 1:
        #     temp = self._natural_statistics(1.)
        # else:
        temp = self._natural_statistics(jnp.zeros(self._sample_space_dim))

        self._params_num = temp.shape[0]
        self._remaining_moments_num = 0

        if remaining_statistics:
            self._remaining_moments = jit(jnp.vectorize(remaining_statistics, signature=self._stats_vect_sign))

            def extended_statistics(x):
                return jnp.concatenate((self._natural_statistics(x), self._remaining_moments(x)))

            self._extended_statistics = jit(jnp.vectorize(extended_statistics, signature=self._stats_vect_sign))

            if sample_space_dimension == 1:
                temp = self._remaining_moments(1.)
            else:
                temp = self._remaining_moments(jnp.ones(self._sample_space_dim))

            self._remaining_moments_num = temp.shape[0]
        else:
            self._extended_statistics = self._natural_statistics
            self._remaining_moments_num = 0

        def log_partition(theta, bijection_params):
            res, max_inner = self.integrate_partition(theta, bijection_params)
            return jnp.log(res) + max_inner

        def log_partition_extended(theta_extended, bijection_params):
            """
            Formula 8.1
            """
            res, max_inner = self.integrate_partition_extended(theta_extended, bijection_params)
            return jnp.log(res) + max_inner

        self._log_partition = jit(jnp.vectorize(log_partition, signature=self._par_vect_sign, excluded=(1,)))
        self._log_partition_jac = jax.jacobian(self._log_partition)
        self._log_partition_hess = jax.hessian(self._log_partition)

        self._log_partition_extended = jit(jnp.vectorize(log_partition_extended,
                                                         signature=self._par_vect_sign, excluded=(1,)))
        self._log_partition_extended_jac = jax.jacobian(self._log_partition_extended)
        self._log_partition_extended_hess = jax.hessian(self._log_partition_extended)

    @property
    def sample_space_dim(self):
        return self._sample_space_dim

    @property
    def params_num(self):
        return self._params_num

    @property
    def remaining_moments_num(self):
        return self._remaining_moments_num

    @property
    def bijection_params(self):
        return self._bijection_params

    @bijection_params.setter
    def bijection_params(self, value):
        self._bijection_params = value

    @property
    def bijection(self):
        return self._bijection

    @property
    def log_dvolume(self):
        return self._log_dvolume

    @property
    def natural_statistics(self):
        return self._natural_statistics

    @property
    def extended_statistics(self):
        return self._extended_statistics

    @property
    def higher_moments(self):
        return self._remaining_moments

    @property
    def log_partition(self):
        return self._log_partition

    @property
    def log_partition_extended(self):
        return self._log_partition_extended

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
    @partial(jit, static_argnums=[0, ])
    def natural_statistics_expectation(self, theta, bijection_params):
        return self._log_partition_jac(theta, bijection_params)

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
    @partial(jit, static_argnums=[0, ])
    def extended_statistics_expectation(self, theta, bijection_params):
        return self._log_partition_extended_jac(jnp.pad(theta, (0, self._remaining_moments_num)), bijection_params)

    @partial(jnp.vectorize, signature='(m)->(m,m)', excluded=[0, 2])
    @partial(jit, static_argnums=[0, ])
    def fisher_metric(self, theta, bijection_params):
        return self._log_partition_hess(theta, bijection_params)

    @partial(jnp.vectorize, signature='(m),(m)->()', excluded=[0, 3])
    @partial(jit, static_argnums=[0, ])
    def bergmann_divergence(self, theta_1, theta_2, bijection_params):
        """
        Compute Bergmann Divergence according to eq. (1.57) Amari 2016.

        Parameters
        ----------
        theta_1
        theta_2
        bijection_params

        Returns
        -------

        """
        psi_1 = self._log_partition(theta_1, bijection_params)
        psi_2 = self._log_partition(theta_2, bijection_params)
        eta_2 = self._log_partition_jac(theta_2, bijection_params)

        return psi_1 - psi_2 - jnp.dot(eta_2, theta_1 - theta_2)

    @abstractmethod
    def get_density_values(self, grid_limits: jnp.ndarray, theta: jnp.ndarray, nb_of_points: jnp.ndarray,
                           bijection_params: tuple) -> \
            tuple[jnp.ndarray, jnp.ndarray]:
        """
        get density values for given `theta` on `grid_limits` as an array of `N_d x 2` and `nb_of_points` as an
        array of `N_d` integers

        Parameters
        ----------
        grid_limits : tuple
        theta : ndarray
        nb_of_points    : int
        bijection_params : tuple

        Returns
        -------
        result  : ndarray, ndarray
            x_grid and the density result.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate_partition(self, theta: jnp.ndarray, bijection_params: tuple) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        This is where the children class need to implement their numerical integration to
        obtain the integration of log partition function at theta in R^d

        Parameters
        ----------
        theta: ndarray
            exponential family natural parameter

        bijection_params: tuple
            bijection parameters

        Returns
        -------
        res: ndarray
            integration result
        max_inner: ndarray
            scaling_factor
        """
        raise NotImplementedError

    @abstractmethod
    def integrate_partition_extended(self, theta_extended: jnp.ndarray, bijection_params: tuple) -> tuple[jnp.ndarray,
    jnp.ndarray]:
        """
        This is where the children class need to implement their numerical integration to
        obtain the integration of log partition extended function at theta in R^d

        Parameters
        ----------
        theta_extended: ndarray
            extended exponential family natural parameter
        bijection_params: tuple
            bijection parameters

        Returns
        -------
        res: ndarray
            integration result
        max_inner: ndarray
            scaling_factor
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(self, fun: Callable[[jnp.ndarray, ], jnp.ndarray], bijection_params: tuple) -> jnp.ndarray:
        """
        TODO: not implemented in all children
        This is where the children class need to implement their numerical integration to
        obtain the expectation of a statistic at theta in R^d

        Parameters
        ----------

        fun: Callable
            function of x, mapping from R^n to R^m
        bijection_params: tuple
            bijection parameters

        Returns
        -------
        res: jnp.ndarray
            integration result
        """
        raise NotImplementedError

    def expected_value(self, statistic: Callable[[jnp.ndarray, ], jnp.ndarray], theta: jnp.ndarray,
                       bijection_params: tuple) -> jnp.ndarray:
        """
        This is where the children class need to implement their numerical integration to
        obtain the expectation of a statistic at theta in R^d

        Parameters
        ----------

        statistic: Callable
            Statistics, function of x, mapping from R^d to R^m
        theta: ndarray
            exponential family's natural parameter
        bijection_params: tuple
            bijection parameters

        Returns
        -------
        res: ndarray
            integration result
        """
        psi = self.log_partition(theta, bijection_params)

        def integrand(x: jnp.ndarray) -> jnp.ndarray:
            """
            Compute s(x)p_theta(x), s:R^d->R^m

            Parameters
            ----------
            x: ndarray
                vector from R^d

            Returns
            -------
            values
            """
            return statistic(x).T * jnp.exp(self.natural_statistics(x) @ theta - psi)

        return self.integrate(integrand, bijection_params)

    def sample(self,
               shape: tuple,
               theta: jnp.ndarray,
               indices: jnp.ndarray,
               bijection_params: tuple,
               key: jnp.ndarray
               ) -> jnp.ndarray:
        """
        Perform importance sampling of from the exponential family with parameter theta

        Parameters
        ----------
        shape: int
            samples count
        theta: ndarray
            parameters
        indices: ndarray
            indices for x_1, ... , x_d, in natural statistics where d is the sample space dimension
        bijection_params: tuple
            bijection parameters
        key: ndarray
            jax.random.PRNGKey instance

        Returns
        -------
        samples: ndarray
            samples of shape (1,count,d)
        """

        # fist get a Gaussian samples from gaussian density with mean and variance according to
        # p_\theta mean and variance
        eta = self.natural_statistics_expectation(theta, bijection_params)
        g = self.fisher_metric(theta, bijection_params)
        indices_grid = tuple(jnp.meshgrid(indices,
                                          indices, indexing='xy'))

        mean = jnp.take(eta, indices)
        cov = g[indices_grid]

        key, a_key = jrandom.split(key)
        gaussian_samples = jrandom.multivariate_normal(key, mean, cov, shape)
        inner = self._natural_statistics(gaussian_samples) @ theta
        max_inner = jnp.max(inner)
        log_exponential_density_un_normalized = inner - max_inner
        log_gaussian = resampling.log_gaussian_density(gaussian_samples - mean, cov)
        log_weight = resampling.normalize_log_weights(log_exponential_density_un_normalized - log_gaussian)

        uni = jrandom.uniform(key)

        samples = resampling.systematic_or_stratified(gaussian_samples, log_weight, uni)

        return samples

    def empirical_kld(self,
                      samples: jnp.ndarray,
                      theta: jnp.ndarray,
                      bijection_params: tuple) -> float:
        """
        Relative entropy from an empirical density with samples, to
        an exponential density with natural parameters theta and bijection_params

        Parameters
        ----------
        samples: jnp.ndarray
            samples from an empirical density
        theta: jnp.ndarray
            natural parameters
        bijection_params: tuple
            bijection parameters

        Returns
        -------
        result: float
            relative entropy.

        """
        psi = self._log_partition(theta, bijection_params)
        N = self._state_history.shape[0]  # sample size
        mean_log_weights = -jnp.log(N) / N
        relative_entropy = mean_log_weights - jnp.mean((self._natural_statistics(samples) @ theta - psi))

        return relative_entropy
