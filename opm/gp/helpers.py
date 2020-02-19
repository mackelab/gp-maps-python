import numpy as np

from opm.gp.prior import prior_covariance
from opm.gp.prior.kernels import mexican_hat_kernel


def get_2d_indices(size):
    """ Given the size of an OPM, compute the indices of the pixels
    
    Args:
        size: size of the orientation preference map, either scalar (square OPM) or tuple (rectangular OPM)
    
    Returns:
        An npixels x 2 matrix, where the kth row contains the x and y coordinates of the kth pixel
    """

    if isinstance(size, int):
        sx, sy = size, size
    else:
        sx, sy = size

    indices = np.array(np.unravel_index(np.arange(sx * sy), dims=(sx, sy))).T

    return indices


def calc_actual_rank(size, atol=1e-7, sigma=6.0, alpha=2.0):
    idx = get_2d_indices(size)
    K = prior_covariance(idx, kernel=mexican_hat_kernel, sigma=sigma, alpha=alpha)

    eigvals = np.linalg.eigvalsh(K)

    return np.where(np.isclose(eigvals[::-1], 0, atol=atol))[0][0]