import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.utils.extmath import fast_logdet


class NoiseModel:
    """ Base class for noise models, defining noise covariance and variance based on the low-rank structure.
    """

    def __init__(self):
        self._covariance = None

    @property
    def covariance(self):
        """ Covariance matrix: D + GG'
        """
        if self._covariance is None:
            self._covariance = self.D + self.G @ self.G.T
        return self._covariance

    @property
    def precision(self):
        raise NotImplementedError

    @property
    def variance(self):
        """ Variance (diagonal of covariance matrix)
        """
        return np.diag(self.covariance)


class FixedNoise(NoiseModel):

    def __init__(self, sigma):
        """ Initialize D as variance times identity and G as zeros

        Args:
            sigma: noise covariance matrix
        """
        super().__init__()

        self.D = sigma

        self.G = np.zeros((self.D.shape[0], 1))

    @property
    def precision(self):
        return np.linalg.inv(self.D)


class LowRankNoise(NoiseModel):

    def __init__(self, method, q):
        """ Initialize method and dimensionality
        
        Args:
            method: noise fitting method (can be either 'factoran' or 'indep')
            q: rank of GG', only needed for factor analysis model
        """
        super().__init__()
        self.method = method
        self.q = q

        self.D = None
        self.G = None

        self._precision = None
        self._precision_logdet = None

    def fit(self, z, noise_variance_init=None, tol=0.01, max_iter=1000, iterated_power=3):
        """ Fit the noise model given the posterior mean

        Args:
            z: (n_trials, n_pixels) residuals
            **noise_kwargs: contains 'method' and 'q'

        Returns:
            sigma, n x n noise covariance matrix
        """

        if self.method == 'factoran':
            # fit factor analysis model
            self.fa = FactorAnalysis(n_components=self.q, noise_variance_init=noise_variance_init,
                                     tol=tol, max_iter=max_iter, iterated_power=iterated_power)
            self.fa.fit(z)
            self.D = np.diag(self.fa.noise_variance_)
            self.G = self.fa.components_.T

        elif self.method == 'indep':
            # pixel variance across trials
            self.D = np.diag(np.var(z, axis=0))
            self.G = np.zeros((self.D.shape[0], 1))

        return self

    @property
    def precision(self):
        if self.method == "factoran":
            if self._precision is None:
                self._precision = self.fa.get_precision()
            return self._precision
        elif self.method == "indep":
            return np.diag(1 / np.diag(self.D))

    @property
    def precision_logdet(self):
        if self._precision_logdet is None:
            self._precision_logdet = fast_logdet(self.precision)
        return self._precision_logdet

    def log_likelihood(self, z):
        """
        Args:
            z: (n_trials, n_pixels) residuals

        Returns:
            log likelihood of the residuals
        """
        N, n = z.shape
        return - 0.5 * ((z * (z @ self.precision)).mean(axis=1) - self.precision_logdet + n * np.log(2 * np.pi))