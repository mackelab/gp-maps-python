import numpy as np


if __name__ == "__main__":
    import scipy.io

    import matplotlib.pyplot as plt

    from opm import ml_opm, plot_opm
    from opm.stimuli import create_stimuli

    from opm.gp import GaussianProcessOPM
    from opm.gp.helpers import get_2d_indices

    seed = 134
    np.random.seed(seed)

    data = scipy.io.loadmat('/data/neuro/opm/ferretdata.mat')
    data = data['dat']  # data is stored in a dict

    R = np.transpose(data[26:, :200, :, :], (2, 3, 0, 1))

    # downsample image (for performance)
    R = R[:, :, ::2, ::2]

    size = R.shape[2:]
    n = size[0] * size[1]
    N = R.shape[0] * R.shape[1]

    angles = [i * np.pi / 4 for i in range(4)] * 2

    # use 95% of the data as the "ground truth"
    # and 5% of the data as real data
    idx = np.array([False for i in range(100)])
    data_repetitions = 10
    gt_repetititions = 100 - data_repetitions
    idx[np.arange(100)[np.random.choice(100, size=data_repetitions, replace=False)]] = True
    R_gt = R[:, ~idx, :, 50:]
    R_data = R[:, idx, :, 50:]

    V_gt = create_stimuli([1.], angles, gt_repetititions)
    V_data = create_stimuli([1.], angles, data_repetitions)

    data_size = R_data.shape[2:]
    data_n = data_size[0] * data_size[1]

    m_gt = ml_opm(R_gt, V_gt)
    plot_opm((m_gt[0] + 1j * m_gt[1]).reshape(data_size), pinwheels=False, shade=False, rmax=80, title='Ground truth')
    plt.show()

    m_data = ml_opm(R_data, V_data)
    plot_opm((m_data[0] + 1j * m_data[1]).reshape(data_size), pinwheels=False, shade=True, rmax=80,
             title='Empirical map on {}% data'.format(data_repetitions))
    plt.show()

    refit = True

    if refit:
        gp = GaussianProcessOPM(indices=get_2d_indices(data_size))
        gp.fit(stimuli=V_data, responses=R_data.reshape(R_data.shape[0], R_data.shape[1], data_n),
               noise='factoran', method='icd', rank=500, verbose=True, noise_kwargs={'iterations': 3, 'q': 1})

        gp.save(fname='pkl/ferret_{}x{}px_{}trials_{}seed.pkl'.format(*data_size, N, seed))
    else:
        fname = 'pkl/ferret_{}x{}px_{}trials_{}seed.pkl'.format(*data_size, N, seed)
        print(f"Loading saved GP from {fname}")
        gp = GaussianProcessOPM.load(fname=fname)

    # mu = m.reshape(3, size)
    result = (gp.mu_post[0] + 1j * gp.mu_post[1]).reshape(data_size)
    plot_opm(result, pinwheels=True, title='GP posterior mean on {}% data'.format(data_repetitions))
    plt.show()

    decode_stimuli = create_stimuli([1.], angles, 1).reshape(8, 3)

    # TODO: refactor the decoding into one method based on the log-likelihood
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

    print("Decoding accuracy: {}".format(np.sum(true == decoded) / true.size))
