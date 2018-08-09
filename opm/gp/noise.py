import numpy as np
from sklearn.decomposition import FactorAnalysis

class FixedNoise:

    def __init__(self, sigma):
        """ Initialize D as variance times identity and G as zeros

        Args:
            sigma: noise covariance matrix
        """

        self.D = sigma

        self.G = np.zeros((self.D.shape[0], 1))


class LowRankNoise:

    def __init__(self, method, q):
        self.method = method
        self.q = q

        self.D = None
        self.G = None

    def fit(self, V, R, mu=None):
        """ Fit the noise model given the posterior mean

                Args:
                    V: stimuli, N_cond x N_rep x d array, stimulus conditions for each trial
                    R: responses, N_cond x N_rep x n_x x n_y array, responses from an experiment
                    mu: posterior mean
                    **noise_kwargs: contains 'method' and 'q'

                Returns:
                    sigma, n x n noise covariance matrix
                """

        d = V.shape[2]
        N = R.shape[0] * R.shape[1]
        n = R.shape[2] * R.shape[3]

        # posterior mean not given
        if mu is None:
            z = R.copy()
            for i, R_i in enumerate(z):
                m_i = R_i.mean(axis=0)
                z[i] = R_i - m_i

            z = z.reshape(N, n)
        else:
            # compute residuals
            z = R.reshape(N, n) - V.reshape(N, d) @ mu.reshape(d, n)

        if self.method == 'factoran':
            # fit factor analysis model
            fa = FactorAnalysis(n_components=self.q)
            fa.fit(z)
            self.D = np.diag(fa.noise_variance_)
            self.G = fa.components_.T

        elif self.method == 'indep':
            # pixel variance across trials
            self.D = np.diag(np.var(z, axis=0))
            self.G = np.zeros((self.D.shape[0], 1))

        return self
