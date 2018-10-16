import numpy as np


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
