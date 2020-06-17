import numpy as np


if __name__ == "__main__":
    import scipy.io

    import matplotlib.pyplot as plt

    from opm import plot_opm
    from opm.stimuli import create_stimuli

    from opm.gp import GaussianProcessOPM

    seed = 134
    np.random.seed(seed)

    data = scipy.io.loadmat('data/imagingdata.mat')
    data = data['dat']  # data is stored in a dict

    R = np.transpose(data[26:, 100:200, :, :], (2, 3, 0, 1))




    angles = [i * np.pi / 4 for i in range(4)] * 2

    # use 95% of the data as the "ground truth"
    # and 5% of the data as real data
    idx = np.load("demos/pkl/idx.npy")

    R_gt = R[:, ~idx, :, :]


    N_cond, N_rep, nx, ny = R_gt.shape
    N = N_cond * N_rep
    n = nx * ny
    size = nx, ny

    V_gt = create_stimuli([1.], angles, N_rep)

    r = R_gt.reshape(N, n)
    V = V_gt.reshape(N, 3)



    m_gt = np.linalg.inv(V.T @ V) @ V.T @ r
    plot_opm((m_gt[0] + 1j * m_gt[1]).reshape(size), pinwheels=False, shade=False, rmax=80, title='Ground truth')
    plt.show()


    # Load the GP that was fit on the other 40 trials
    fname = 'demos/pkl/ferret_{}x{}px_{}trials_{}seed.pkl'.format(*size, 40, seed)
    print(f"Loading saved GP from {fname}")
    gp = GaussianProcessOPM.load(fname=fname)

    # mu = m.reshape(3, size)
    result = (gp.mu_post[0] + 1j * gp.mu_post[1]).reshape(size)
    plot_opm(result, pinwheels=True, title="GP posterior mean")
    plt.show()

    # Decode the ground truth stimuli using the GP

    decode_stimuli = create_stimuli([1.], angles, 1).reshape(8, 3)

    def log_likelihood(gp, response, stimuli):
        """ Computes the log likelihood of a response pattern under a set of stimuli (eqn 9 in the NeuroImage paper)

        Args:
            response (np.array): n x 1 response pattern
            stimuli (np.array): d x N set of N stimuli

        Returns:
            ll (np.array): N x 1 log likelihood for each stimulus
        """
        mu_post = gp.mu_post.reshape(3, response.shape[0])
        error = response - (mu_post.T @ stimuli.T)
        ll = - 0.5 * error.T @ gp.noise.inverse_covariance @ error

        return np.diag(ll)

    # gp.log_likelihood(r[0][:,np.newaxis], decode_stimuli)

    """# TODO: refactor the decoding into one method based on the log-likelihood
    true = np.zeros((8, gt_repetititions))
    decoded = np.zeros((8, gt_repetititions))
    for i in range(8):
        for j in range(gt_repetititions):
            ll = gp.log_likelihood(R_gt[i, j].reshape(data_n, 1), decode_stimuli)
            true_stim = V_gt[i, j]

            true_ij = np.rad2deg(np.arctan2(true_stim[1], true_stim[0]) / 2)
            decoded_ij = np.rad2deg(angles[np.argmax(ll)])
            true[i, j] = true_ij % 180
            decoded[i, j] = decoded_ij % 180

    print("Decoding accuracy: {}".format(np.sum(true == decoded) / true.size))"""
