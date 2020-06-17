import numpy as np
import inspect

from opm.gp.prior.kernels import mexican_hat_kernel
from opm.gp.prior.cholesky import ridge_cholesky, incomplete_cholesky
from opm.gp.prior.match_radial_component import match_radial_component


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

    def __init__(self, idx, kernel, method="icd", ridge=1e-6):
        """ Setup LowRank prior
        Args:
            method: currently allows 'icd' or None, defaults to None
            rank: rank of the approximation (not used when None is specified)
        """

        if method not in ["full", "icd"]:
            raise ValueError('No valid low-rank method specified.')

        self.method = method
        self.x = idx
        self.kernel = kernel
        self.ridge = ridge

        self._fit = False

    def init_from_empirical(self, m, p0=None, verbose=False):
        """ Estimate the prior hyperparameters by matching them to the radial component
            of the empirical map (see match_radial_components).

        Args:
            m: d x nx x ny, empirical map
            p0: (dict) initial guess for the kernel hyperparameters

        Returns:
            self.kernel_params, dict containing the names and optimized values of the hyperparameters
        """

        if verbose:
            print('Estimating prior hyperparameters:')

        # get names and default values for hyperparameters
        s = inspect.signature(self.kernel)
        hyperparams = list(s.parameters.values())[2:]
        if not p0:
            p0 = {p.name: p.default for p in hyperparams}

        p_opt = match_radial_component(m=m, p0=p0)

        kernel_params = {p.name: val for p, val in zip(hyperparams, p_opt)}

        if verbose:
            print(kernel_params)

        if verbose:
            print("Recomputing prior")

        self.recompute(**kernel_params)

        return self.kernel_params

    @property
    def is_fit(self):
        return self._fit

    def recompute(self, **kernel_kwargs):
        """ Learn a (low-rank) prior.
        Args:
            kernel_kwargs: get passed to kernel(...)
            
        After fitting, the prior has the following attributes:
            - G: Low rank approximation K = GG' of prior covariance matrix
            - K: Full prior covariance matrix (depending on low-rank method)
        """

        self.kernel_params = kernel_kwargs

        if not self.method:
            self.K = prior_covariance(self.x, kernel=self.kernel, **kernel_kwargs)
            self.G = ridge_cholesky(self.K)

            self.D = np.eye(self.x.shape[0]) * self.ridge

        elif self.method.lower() == 'icd':
            G = incomplete_cholesky(self.x, eta=1e-4, kernel=self.kernel, ridge=0, **kernel_kwargs)["R"]
            self.G = G.T

            self.rank = self.G.shape[1]

            n = self.x.shape[0]

            # correct the diagonal
            prior_var_uncorrected = np.diag(self.G @ self.G.T)
            prior_var_exact = np.ones(n) * self.kernel(0, 0, **kernel_kwargs)

            self.D = np.diag(prior_var_exact - prior_var_uncorrected + 2 * self.ridge)

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

    def sample(self, sample_shape):
        return np.random.multivariate_normal(np.zeros(self.K.shape[0]), self.K, size=sample_shape)
