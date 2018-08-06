from .kernels import mexican_hat_kernel, fixed_k_mexhat
from .helpers import get_2d_indices
from .match_radial_component import match_radial_component
from .linalg import ICD

from ..opm import calculate_map

import numpy as np
import inspect
import dill as pickle
import os
from sklearn.decomposition import FactorAnalysis

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


class LowRankPrior():
    """ (Low rank approximation of a) covariance matrix. Can be fit using Inverse Cholesky Decomposition (icd),
        or without a low-rank method.
    """
    
    def __init__(self, x, method=None, rank=None):
        """ Setup LowRank prior
        Args:
            method: currently allows 'icd' or None, defaults to None
            rank: rank of the approximation (not used when None is specified)
        """
        
        if method not in [None, 'icd']:
            raise ValueError('No valid low-rank method specified.')
        self.method = method
        self.x = x
        
        if not rank:
            self.rank = x.shape[0]
        else:
            self.rank = rank
            
        
            
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
            icd = ICD(rank=self.rank)
            icd.fit(self.x, kernel, ridge, **kernel_kwargs)
            self.G = icd.G
            self.K = icd
            
            n = self.x.shape[0]
            
            # correct the diagonal
            prior_var_uncorrected = np.zeros(n)
            prior_var_exact = np.zeros(n)

            for k in range(n):
                prior_var_uncorrected[k] = self.G[k,:] @ self.G[k,:].T
                prior_var_exact[k] = kernel(self.x[k], self.x[k], **kernel_kwargs)

            self.D = np.diag(prior_var_exact - prior_var_uncorrected + 2 * ridge)
            

    def __getitem__(self, idx):
        """ Access portions of K = GG'
        Args:
            item: indices to access K = GG'
        Return:
            K[idx[0], idx[i]]
        """

        return self.K[idx[0], idx[1]]

    
    
class GaussianProcessOPM():
    """ A Gaussian process used to infer an orientation preference map (OPM) from imaging data.
    """
    
    def __init__(self, size, prior_rank, prior_method='icd', kernel=fixed_k_mexhat):
        """ Initialize prior fitting method and dimensionalities
        
        Args:
            size: tuple (x, y) or int (results in square map)
            prior_rank: rank of low-rank prior_approximation (only used if prior_method is given)
            prior_method: can be either 'icd' or None
            kernel: kernel function of structure f(x, y, **hyperparams). 
                    defaults to mexican hat with sigma and and alpha as parameters and fixed k.
        """
        self.size = size
        self.idx = get_2d_indices(size)
        
        
        self.rank = prior_rank
        self.prior_method = prior_method
        
        self.kernel = kernel
       
    
    def fit_prior(self):
        """ Learn a (low-rank) represenation of the prior covariance.
        
        Return:
            self.prior (fitted LowRankPrior object)
        """
        self.prior = LowRankPrior(self.idx, method=self.prior_method, rank=self.rank)
        self.prior.fit(kernel=self.kernel, **self.kernel_params)
        return self.prior
        
    
    def optimize(self, stimuli, responses, verbose=False):
        """ Estimate the prior hyperparameters by matching them to the radial component 
            of the empirical map (see match_radial_components).
            
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n_x x n_y array, responses from an experiment 
        
        Returns:
            self.kernel_params, dict containing the names and optimized values of the hyperparameters
        """
        
        # get names and default values for hyperparameters
        s = inspect.signature(self.kernel)
        hyperparams = list(s.parameters.values())[2:]
        p0 = {p.name: p.default for p in hyperparams}
        
        p_opt = match_radial_component(responses, stimuli, p0=p0)
        
        self.kernel_params = {p.name: val for p, val in zip(hyperparams, p_opt)}
        
        if verbose:
            print(self.kernel_params)
        
        return self.kernel_params
    
    
    def fit_posterior(self, stimuli, responses, noise_cov):
        """ Given a set of stimuli and responses, compute the posterior mean and covariance
        
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n_x x n_y array, responses from an experiment
            noise_cov: n x n array, noise covariance matrix
        
        Returns:
            self.mu_post, self.K_post: posterior mean and covariance
        """
        V = stimuli
        N = stimuli.shape[0] * stimuli.shape[1]
        d = stimuli.shape[2]
        
        V = V.reshape((N,d))
        
        nx = responses.shape[2]
        ny = responses.shape[3]
        n = nx * ny
        R = responses.reshape((N,n))
        
        G = self.prior.G
        K = G @ G.T + self.prior.D
        beta = 2 / R.shape[0] 
        
        S = np.linalg.inv(noise_cov)
        
        # TODO: what's happening here that is not working with non-square OPMs?
        
        K_post_c =  K - 1/beta * K @ (S - S @ G @ np.linalg.inv(beta * np.eye(self.rank) + G.T @ S @ G) @ G.T @ S) @ K
        
        self.K_post = np.kron(np.eye(d), K_post_c)
        
        # inefficient version (keeping the comment for readability)
        # K_post = np.linalg.inv(np.linalg.inv(K_m) + np.kron(N/2 * np.eye(d), K_e))
        
        # different way of writing vector averaging (calculate_map, i.e. max likelihood)
        # vr = np.zeros((n*d,1))
        # for v, r in zip(V, R):
        #    vr += np.kron(v, r)[:,np.newaxis]
        
        # TODO: this can be made more efficient by leveraging the low-rank stuff

        self.mu_post = self.K_post @ np.kron(np.eye(d), S) @ calculate_map(responses, stimuli).reshape(n * d, 1)
        
        return self.mu_post, self.K_post
    
    
    def learn_noise_model(self, V, R, mu, **noise_kwargs):
        d = V.shape[2]
        N = R.shape[0] * R.shape[1]
        n = R.shape[2] * R.shape[3]
        
        # compute residuals
        z = R.reshape(N, n) - V.reshape(N, d) @ mu.reshape(d, n)
        
        if noise_kwargs['method'] == 'factoran':
            # fit factor analysis model
            fa = FactorAnalysis(n_components=noise_kwargs['q'])
            fa.fit(z)
            sigma = fa.get_covariance()
            
        elif noise_kwargs['method'] == 'indep':
            # pixel variance across trials
            sigma = np.diag(np.var(z, axis=0))
        
        return sigma
        
        
    def fit(self, stimuli, responses, noise='factoran', noise_kwargs=None, verbose=False):
        """ Complete fitting procedure:
            - Estimate prior hyperparameters using empirical map
            - Fit prior covariance
            - Compute the posterior mean and covariance (assuming a given noise covariance)
        
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n_x x n_y array, responses from an experiment 
            noise_cov: can be
                        - n x n numpy.ndarray, given noise covariance matrix 
                        - 'factoran': iterative factor analysis noise estimation
                        - 'indep': iterative independent noise estimation
            verbose: boolean, do you want to print progress info?
        
        Returns:
            self.mu_post: posterior mean
        """
        
        # check if valid noise estimation method is specified
        if not (type(noise) is np.ndarray or (type(noise) is str and noise in ['factoran', 'indep'])):
            raise ValueError("Please specify a valid noise model.")
            
        # get dimensionalities
        d = stimuli.shape[2]
        N_cond = responses.shape[0]
        N_rep = responses.shape[1]
        N = N_cond * N_rep
        n = responses.shape[2] * responses.shape[3]
        
        
        if verbose:
            print('*** Estimating prior hyperparameters ***')
            
        self.optimize(stimuli, responses, verbose=verbose)
        
        
        if verbose:
            print('*** Fitting prior ***')
            
        self.fit_prior()
        
        
        if verbose:
            print('*** Fitting posterior ***')
            
        if type(noise) is np.ndarray:
            # given noise covariance matrix
            self.fit_posterior(stimuli, responses, noise)
            
            
        else:
            if noise_kwargs is None:
                noise_kwargs = {}
                
            # default noise model parameters
            noise_kwargs.setdefault('iterations', 3)
            noise_kwargs.setdefault('q', 2)
            noise_kwargs.setdefault('method', noise)
            
            # compute initial estimate, assuming the whole signal is noise
            sigma_noise_init = np.zeros((n, n))
            for R_i in responses.reshape(N_cond, N_rep, -1):
                C_i = np.cov(R_i.T)
                sigma_noise_init += C_i

            sigma_noise_init /= N_cond
            
            # iterative noise fitting procedure
            for i in range(noise_kwargs['iterations']):
                
                if verbose:
                    print('Fitting noise model: iteration {}'.format(i+1))
                
                if i == 0:
                    # in the first step use the initial estimate
                    mu, _ = self.fit_posterior(stimuli, responses, sigma_noise_init)
                    
                # learn the noise model (either indep or factoran) given current posterior mean
                sigma_noise = self.learn_noise_model(V=stimuli, R=responses, mu=mu, **noise_kwargs)
                
                # get updated estimate of posterior mean using current estimate of noise covariance
                mu, _ = self.fit_posterior(stimuli, responses, sigma_noise)
            
            self.noise_cov = sigma_noise
        
        
        return self.mu_post
    
    
    
    def save(self, fname):
        if not isinstance(fname, str):
            raise ValueError("Parameter fname has to be a string.")

        try:
            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname), exist_ok=True)

            pickle.dump(self, open(fname, "wb"))
        except IOError as io:
            print("IOError while saving class: {}".format(io))

    @staticmethod
    def load(fname):
        if not isinstance(fname, str):
            raise ValueError("Parameter fname has to be a string.")

        try:
            sampler = pickle.load(open(fname, "rb"))
        except IOError as io:
            print("IOError while saving class: {}".format(io))

        return sampler
        
    
