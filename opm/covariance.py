import numpy as np

from opm.gp.helpers import get_2d_indices


def cov2corr(cov):
    std = np.sqrt(np.diag(cov))[:, np.newaxis]
    return cov / std / std.T


def avg_neighbor_corr(cov, size, radius=2):
    # convert noise covariance to correlation matrix
    # noise_corr = cov2corr(gp.noise.covariance)

    # initialize empty image
    neighbor_corr = np.zeros(size)

    idx = get_2d_indices(size)

    # for each index
    # i is the 1d index
    # ix is the 2d (x, y) index
    for i, ix in enumerate(idx):
        # compute euclidean distance from this index to all others
        dist = np.sqrt(np.sum(np.square(ix - idx), axis=1))

        # keep all indices smaller than some distance
        neighbors = np.where(dist <= radius)[0]
        # remove the index itself
        neighbors = neighbors[neighbors != i]

        # compute average correlation between i and the neighbors
        neighbor_corr[ix[0], ix[1]] = np.mean(
            cov[i, neighbors] / np.sqrt(cov[i, i]) / np.sqrt(cov[neighbors, neighbors]))

    return neighbor_corr
