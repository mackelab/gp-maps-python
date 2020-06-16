import numpy as np
from opm.gp.prior.kernels import mexican_hat_kernel
from opm.gp.prior.cholesky import ridge_cholesky, incomplete_cholesky


def prior_covariance(idx, kernel, **kwargs):
    """ Compute the prior covariance matrix for an OPM, given a kernel function
    Args:
        idx: The indices at which to compute the covariance (npixels x 2 matrix, where the kth column contains the x and y coordinates of the kth pixel)
        kernel: a kernel function that takes two vectors x and y
        kwargs: parameters for the kernel function
        
    Returns:
        a matrix of dimensionality npixels x npixels
    """

    K = np.zeros((idx.shape[0], idx.shape[0]))
    for i in range(idx.shape[0]):
        for j in range(idx.shape[0]):
            K[i, j] = kernel(idx[i], idx[j], **kwargs)

    return K


class LowRankPrior:
    """ (Low rank approximation of a) covariance matrix. Can be fit using Inverse Cholesky Decomposition (icd),
        or without a low-rank method.
    """

    def __init__(self, x, method=None):
        """ Setup LowRank prior
        Args:
            method: currently allows 'icd' or None, defaults to None
            rank: rank of the approximation (not used when None is specified)
        """

        if method not in [None, 'icd']:
            raise ValueError('No valid low-rank method specified.')
        self.method = method
        self.x = x

        self._fit = False

    @property
    def is_fit(self):
        return self._fit

    def fit(self, kernel=mexican_hat_kernel, ridge=1e-4, **kernel_kwargs):
        """ Learn a (low-rank) prior.
        Args:
            kernel: a kernel function that takes two vectors x and y
            ridge: element that gets added to the diagonal to avoid numerical instabilites (use depends on low-rank
                    method)
            kernel_kwargs: get passed to kernel(...)
            
        After fitting, the prior has the following attributes:
            - G: Low rank approximation K = GG' of prior covariance matrix
            - K: Full prior covariance matrix (depending on low-rank method)
        """

        if not self.method:
            self.K = prior_covariance(self.x, kernel=kernel, **kernel_kwargs)
            self.G = ridge_cholesky(self.K)

            self.D = np.eye(self.x.shape[0]) * ridge

        elif self.method.lower() == 'icd':
            G = incomplete_cholesky(self.x, eta=1e-4, kernel=kernel, **kernel_kwargs)["R"]
            self.G = G.T

            self.rank = self.G.shape[1]

            n = self.x.shape[0]

            # correct the diagonal
            prior_var_uncorrected = np.zeros(n)
            prior_var_exact = np.zeros(n)

            prior_var_uncorrected = self.G @ self.G.T
            prior_var_exact = np.ones(n) * kernel(0, 0, **kernel_kwargs)

            self.D = np.diag(prior_var_exact - prior_var_uncorrected + 2 * ridge)

            self.K = self.G @ self.G.T + self.D

        self._fit = True

    def __getitem__(self, idx):
        """ Access portions of K = GG'
        Args:
            item: indices to access K = GG'
        Return:
            K[idx[0], idx[i]]
        """

        return self.K[idx[0], idx[1]]
