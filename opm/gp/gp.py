from opm.gp.prior import kernels, LowRankPrior
from opm.gp.noise import LowRankNoise
from opm.gp.lowrank import calc_postmean

from opm import ml_opm

import numpy as np
import dill as pickle
import os


class GaussianProcessOPM:
    """ A Gaussian process used to infer an orientation preference map (OPM) from imaging data.
    """

    def __init__(self, prior: LowRankPrior, noise: LowRankNoise):
        """ Initialize prior fitting method and dimensionalities
        
        Args:
            prior: LowRankPrior instance
        """

        self.prior = prior
        self.noise = noise

        self.K_post = None
        self.mu_post = None

    def fit(self, stimuli, responses, noise_kwargs=None, verbose=False, calc_postcov=False):
        """ Complete fitting procedure:
            - Estimate prior hyperparameters using empirical map
            - Fit prior covariance
            - Compute the posterior mean and covariance (assuming a given noise covariance)
        
        Args:
            stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
            responses: N_cond x N_rep x n array, responses from an experiment
            noise_cov:  - 'factoran': iterative factor analysis noise estimation
                        - 'indep': iterative independent noise estimation
            verbose: boolean, do you want to print progress info?
        
        Returns:
            self.mu_post: posterior mean
        """

        # get dimensionalities of stimulus matrix
        N_cond = responses.shape[0]
        N_rep = responses.shape[1]
        d = stimuli.shape[2]
        N = N_cond * N_rep

        V = stimuli.reshape(N, d)
        beta = 1 / np.diag(V.T @ V).mean()

        nx = responses.shape[2]
        ny = responses.shape[3]
        n = nx * ny

        r = responses.reshape(N, n)

        if not self.prior.is_fit:
            m_emp = np.linalg.inv(V.T @ V) @ V.T @ r
            self.prior.init_from_empirical(m_emp.reshape(d, nx, ny), verbose=verbose)

        if verbose:
            print('*** Fitting posterior ***')

        if noise_kwargs is None:
            noise_kwargs = dict()

        # default noise model parameters
        noise_kwargs.setdefault('iterations', 3)
        noise_kwargs.setdefault('max_iter', 1000)
        noise_kwargs.setdefault('tol', 0.01)
        noise_kwargs.setdefault('iterated_power', 3)

        n_iter = noise_kwargs['iterations']
        # iterative noise fitting procedure
        for i in range(n_iter):

            if verbose:
                print('Fitting noise model: iteration {} / {}'.format(i + 1, n_iter))

            if i == 0:
                noise_var_init = None
                # for i==0, the posterior mean is None, thus we assume all the signal is noise
                z = (responses - responses.mean(axis=1, keepdims=True)).reshape(N, n)
            else:
                noise_var_init = self.noise.variance

                # compute residuals
                z = r - V @ self.mu_post

            # learn the noise model
            self.noise.fit(z=z, noise_variance_init=noise_var_init,
                           max_iter=noise_kwargs['max_iter'], tol=noise_kwargs['tol'],
                           iterated_power=noise_kwargs['iterated_power'])

            # get updated estimate of posterior mean using current estimate of noise covariance
            if i == noise_kwargs['iterations'] - 1 and calc_postcov:
                G = self.prior.G
                K = G @ G.T + self.prior.D

                S = np.linalg.inv(self.noise.covariance)
                K_post_c = K - 1 / beta * K @ (
                        S - S @ G @ np.linalg.inv(beta * np.eye(self.prior.rank) + G.T @ S @ G) @ G.T @ S) @ K

                self.K_post = np.kron(np.eye(d), K_post_c)
                # calculate the full posterior covariance (only in the last iteration)

            # self.mu_post = np.kron(np.eye(d), K_post_c @ S) @ mhat

            # inefficient version (keeping the comment for readability)
            # K_post = np.linalg.inv(np.linalg.inv(K_m) + np.kron(N/2 * np.eye(d), K_e))

            # different way of writing vector averaging (calculate_map, i.e. max likelihood)
            # vr = np.zeros((n*d,1))
            # for v, r in zip(V, R):
            #    vr += np.kron(v, r)[:,np.newaxis]

            if verbose:
                print('Computing posterior mean: iteration {} / {}'.format(i + 1, n_iter))

            # use the low-rank approximations of prior and noise to compute the posterior mean (see lowrank.py)
            self.mu_post = calc_postmean(r.T @ V, beta=beta, prior=self.prior, noise=self.noise).T

        return self.mu_post, self.K_post

    def log_likelihood(self, response, stimuli):
        """ Computes the log likelihood of a response pattern under a set of stimuli (eqn 9 in the NeuroImage paper)

        Args:
            response (np.array): n x 1 response pattern
            stimuli (np.array): d x N set of N stimuli

        Returns:
            ll (np.array): N x 1 log likelihood for each stimulus
        """
        z = response - stimuli @ self.mu_post
        ll = self.noise.log_likelihood(z)

        return ll

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
            print("IOError while loading file: {}".format(io))

        return gp


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from opm import make_opm, plot_opm
    from opm.response import compute_responses
    from opm.stimuli import create_stimuli
    from opm.gp.helpers import get_2d_indices

    size = (200, 200)

    n = size[0] * size[1]
    d = 3

    idx = get_2d_indices(size)

    # ground truth opm
    m = make_opm(size, alpha=2, k=2, sigma=10, d=d)

    f, ax, _ = plot_opm(m[0] + 1j * m[1], shade=True)
    plt.show()

    # compute responses
    contrasts = [1, 0.5]
    orientations = [i * np.pi / 8 for i in range(8)]
    repetitions = 8

    stim = create_stimuli(contrasts, orientations, repetitions)

    R = compute_responses(m, stim, noise=0.5)

    V = stim.reshape(stim.shape[0] * stim.shape[1], d)

    mhat = ml_opm(R, stim)
    mhat = mhat[0] + 1j * mhat[1]
    plot_opm(mhat, title="Empirical map")
    plt.show()

    prior = LowRankPrior(idx=idx, method="icd", kernel=kernels.fixed_k_mexhat)

    noise = LowRankNoise(method="factoran", q=2)

    gp = GaussianProcessOPM(prior=prior, noise=noise)

    mu_post, K_post = gp.fit(stimuli=stim, responses=R, verbose=True, calc_postcov=False)

    f, ax, _ = plot_opm((mu_post[0] + 1j * mu_post[1]).reshape(size), title='GP posterior mean',
                        shade=True)
    plt.show()

    noise_var = np.diag(gp.noise.covariance)

    # plt.imshow(np.log10(noise_var).reshape(size), cmap='jet')
    # plt.show()
