from opm.gp.prior.match_radial_component import match_radial_component
from opm.gp.prior import LowRankPrior, kernels
from opm.gp.noise import FixedNoise, LowRankNoise
from opm.gp.lowrank import calc_postmean

from opm import ml_opm

import numpy as np
import inspect
import dill as pickle
import os


class GaussianProcessOPM:
    """ A Gaussian process used to infer an orientation preference map (OPM) from imaging data.
    """

    def __init__(self, indices, kernel=kernels.fixed_k_mexhat):
        """ Initialize prior fitting method and dimensionalities
        
        Args:
            indices: n x 2 array of indices
        """
        self.idx = indices

        self.kernel = kernel
        self.kernel_params = {}

        self.prior = None
        self.noise = None

        self.K_post = None
        self.mu_post = None

    def fit_prior(self, rank=None, method='icd', verbose=False):
        """ Learn a (low-rank) represenation of the prior covariance.

        Return:
            self.prior (fitted LowRankPrior object)
        """

        if not self.prior:
            self.prior = LowRankPrior(self.idx, method=method, rank=rank)

        if not self.kernel_params:
            raise RuntimeError("Prior parameters must be set (or fit) first")

        if verbose:
            print('Calculating the prior from scratch..')

        self.prior.fit(kernel=self.kernel, **self.kernel_params)
        return self.prior

    def optimize(self, stimuli, responses, p0=None, verbose=False):
        """ Estimate the prior hyperparameters by matching them to the radial component 
            of the empirical map (see match_radial_components).
            
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n array, responses from an experiment
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

        p_opt = match_radial_component(responses=responses, stimuli=stimuli, p0=p0)

        self.kernel_params = {p.name: val for p, val in zip(hyperparams, p_opt)}

        if verbose:
            print(self.kernel_params)

        return self.kernel_params

    def fit_posterior(self, stimuli, responses, calc_postcov=False):
        """ Given a set of stimuli and responses, compute the posterior mean and covariance
        
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n array, responses from an experiment
        
        Returns:
            self.mu_post, self.K_post: posterior mean and covariance (K_post is None if not calc_postcov)
        """

        N = stimuli.shape[0] * stimuli.shape[1]
        d = stimuli.shape[2]

        nx = responses.shape[2]
        ny = responses.shape[3]
        n = nx * ny

        # calculate empirical map
        mhat = ml_opm(responses, stimuli).reshape((d, n)).T

        if calc_postcov:
            G = self.prior.G
            K = G @ G.T + self.prior.D
            beta = 2 / N

            S = np.linalg.inv(self.noise.covariance)
            K_post_c = K - 1 / beta * K @ (
                    S - S @ G @ np.linalg.inv(beta * np.eye(self.prior.rank) + G.T @ S @ G) @ G.T @ S) @ K

            self.K_post = np.kron(np.eye(d), K_post_c)

        # self.mu_post = np.kron(np.eye(d), K_post_c @ S) @ mhat

        # inefficient version (keeping the comment for readability)
        # K_post = np.linalg.inv(np.linalg.inv(K_m) + np.kron(N/2 * np.eye(d), K_e))

        # different way of writing vector averaging (calculate_map, i.e. max likelihood)
        # vr = np.zeros((n*d,1))
        # for v, r in zip(V, R):
        #    vr += np.kron(v, r)[:,np.newaxis]

        # use the low-rank approximations of prior and noise to compute the posterior mean (see lowrank.py)
        self.mu_post = calc_postmean(mhat, N, prior=self.prior, noise=self.noise).T
        self.mu_post = self.mu_post.reshape((d, nx, ny))

        return self.mu_post, self.K_post

    def fit(self, stimuli, responses, rank=None, noise='factoran', noise_kwargs=None, verbose=False, calc_postcov=False,
            **prior_kwargs):
        """ Complete fitting procedure:
            - Estimate prior hyperparameters using empirical map
            - Fit prior covariance
            - Compute the posterior mean and covariance (assuming a given noise covariance)
        
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n array, responses from an experiment
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

        if prior_kwargs is None:
            prior_kwargs = {}

        prior_kwargs.setdefault('method', 'icd')
        prior_kwargs.setdefault('p0', None)

        if not self.prior:
            self.prior = LowRankPrior(self.idx, method=prior_kwargs['method'], rank=rank)

        if verbose:
            print('*** Fitting prior ***')

        if not self.kernel_params:
            self.optimize(stimuli, responses, p0=prior_kwargs['p0'], verbose=verbose)

        if not self.prior.is_fit:
            self.fit_prior(verbose=verbose)

        elif verbose:
            print('Using previously fit prior..')

        if verbose:
            print('*** Fitting posterior ***')

        if type(noise) is np.ndarray:
            self.noise = FixedNoise(noise)
            # given noise covariance matrix
            self.fit_posterior(stimuli, responses, calc_postcov)

        else:
            if noise_kwargs is None:
                noise_kwargs = {}

            # default noise model parameters
            noise_kwargs.setdefault('iterations', 3)
            noise_kwargs.setdefault('q', 2)
            noise_kwargs.setdefault('method', noise)
            noise_kwargs.setdefault('max_iter', 1000)
            noise_kwargs.setdefault('tol', 0.01)
            noise_kwargs.setdefault('iterated_power', 3)

            # iterative noise fitting procedure
            for i in range(noise_kwargs['iterations']):

                if verbose:
                    print('Fitting noise model: iteration {}'.format(i + 1))

                if i >= 1:
                    noise_var_init = self.noise.variance
                else:
                    noise_var_init = None

                self.noise = LowRankNoise(method=noise_kwargs['method'], q=noise_kwargs['q'])

                # learn the noise model (either indep or factoran) given the posterior mean
                # for i==0, the posterior mean is None, thus we assume all the signal is noise
                self.noise.fit(V=stimuli, R=responses, mu=self.mu_post, noise_variance_init=noise_var_init,
                               max_iter=noise_kwargs['max_iter'], tol=noise_kwargs['tol'],
                               iterated_power=noise_kwargs['iterated_power'])

                # get updated estimate of posterior mean using current estimate of noise covariance
                if i == noise_kwargs['iterations'] - 1 and calc_postcov:
                    # calculate the full posterior covariance (only in the last iteration)
                    mu, _ = self.fit_posterior(stimuli, responses, calc_postcov)
                else:
                    mu, _ = self.fit_posterior(stimuli, responses)

        return self.mu_post, self.K_post

    def log_likelihood(self, response, stimuli):
        """ Computes the log likelihood of a response pattern under a set of stimuli (eqn 9 in the NeuroImage paper)

        Args:
            response (np.array): n x 1 response pattern
            stimuli (np.array): d x N set of N stimuli

        Returns:
            ll (np.array): N x 1 log likelihood for each stimulus
        """
        error = response - (self.mu_post.T @ stimuli.T)
        ll = - 0.5 * error.T @ self.noise.inverse_covariance @ error

        return np.diag(ll)

    def save(self, fname):
        """ Save this object to a file

        Args:
            fname: file name
        """
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
        """ Load a GP from a file

        Args:
            fname: file name

        Returns:
            a GaussianProcessOPM object
        """
        if not isinstance(fname, str):
            raise ValueError("Parameter fname has to be a string.")

        try:
            gp = pickle.load(open(fname, "rb"))
        except IOError as io:
            print("IOError while saving class: {}".format(io))

        return gp


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from opm import make_opm, plot_opm
    from opm.response import compute_responses
    from opm.stimuli import create_stimuli
    from opm.gp.helpers import get_2d_indices

    size = (50, 50)

    n = size[0] * size[1]
    d = 3

    # ground truth opm
    m = make_opm(size=size, sigma=3., k=2., alpha=2.)

    f, ax, _ = plot_opm(m)
    plt.show()

    idx = get_2d_indices(size)

    # compute responses
    contrasts = [1.]
    orientations = [i * np.pi / 4 - np.pi / 2 for i in range(4)]
    repetitions = 16

    V = create_stimuli(contrasts, orientations, repetitions)

    # number of trials
    N_cond = V.shape[0]
    N = N_cond * V.shape[1]

    R = compute_responses(m, contrasts, orientations, repetitions, sigma=0.1)

    gp = GaussianProcessOPM(indices=idx, kernel=kernels.fixed_k_mexhat)

    mu_post, K_post = gp.fit(stimuli=V, responses=R, method='icd', rank=400, noise='factoran',
                             verbose=True, calc_postcov=True)

    result = mu_post[0] + 1j * mu_post[1]
    result = result.reshape(size)

    f, ax, _ = plot_opm(result, title='GP posterior mean')
    plt.show()

    noise_var = np.diag(gp.noise.covariance)

    # plt.imshow(np.log10(noise_var).reshape(size), cmap='jet')
    # plt.show()
