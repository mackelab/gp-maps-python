import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import filters


def make_opm(size, sigma=4., k=2., alpha=1.):
    """ Generate an orientation preference map (to be used as a fake ground truth). 
     
    
    Args:
        size: dimensionality of the OPM is size x size
        sigma: std of the first gaussian filter
        k: k * sigma is the std of the second gaussian filter
        
    Returns:
        m: complex np.array with shape (size, size)
    """

    if isinstance(size, int):
        sx, sy = size, size
    else:
        sx, sy = size

    # generate white noise for real and imaginary map
    a = np.random.randn(sx, sy)
    b = np.random.randn(sx, sy)

    # apply difference of Gaussians filter to both maps
    a = alpha * filters.gaussian_filter(a, sigma) - alpha * filters.gaussian_filter(a, k * sigma)
    b = alpha * filters.gaussian_filter(b, sigma) - alpha * filters.gaussian_filter(b, k * sigma)

    # combine real and imaginary parts
    m = a + 1j * b

    return m


def calculate_map(responses, stimuli):
    """ Compute OPM components from an experiment (max likelihood solution)
    
    Args:
        stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
        responses: N_cond x N_rep x n_x x n_y array, responses from an experiment 
        
    Returns: estimated map components: d x n_x x n_y array 
    """

    V = stimuli
    d = V.shape[2]
    R = responses
    N_c, N_r, nx, ny = R.shape
    N = N_c * N_r
    n = int(R.size / N)

    V = V.reshape((N, d))

    # least squares estimate of real and imaginary components of the map
    M_flat = np.linalg.inv(V.T @ V) @ V.T @ R.reshape((N, n))

    # reshape map into three
    M = np.zeros((d, nx, ny))
    for i, a in enumerate(M_flat):
        M[i] = a.reshape(nx, ny)

    return M
