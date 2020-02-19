import numpy as np


def response(m, cs, theta, c=0.0, sigma=0.1):
    """ Given an orientation preference map and stimulus parameters, compute a response map
    
    Args:
        m: an OPM
        cs: stimulus contrast (scalar)
        theta: stimulus orientation (scalar)
        c: constant to be added
        sigma: noise standard deviation
        
    Returns:
        An array of the same dimension as r containing responses at each pixel
    """
    # r(x,s) = a(x) * s_1(x) + b(x) * s_2(x) + c + noise
    response = np.real(m) * cs * np.cos(2 * theta) + np.imag(m) * cs * np.sin(2 * theta) + c
    noise = np.random.randn(*m.shape) * sigma
    return response + noise


def compute_responses(m, contrasts, orientations, repetitions, sigma=0.01):
    """ Compute matrix of responses
    
    Args:
        m: an OPM
        contrasts: list of contrast conditions
        orientations: list of orientation conditions (radians)
        repetitions: number of repetitions (integer)
        
    Returns: 
        ntrials x npixels matrix containing responses at each pixel for each trial
    """

    N = len(contrasts) * len(orientations)

    i = 0
    responses = np.zeros((N, repetitions, *m.shape))
    # for every combination
    for theta in orientations:
        for c in contrasts:
            for j in range(repetitions):
                responses[i, j, :, :] = response(m, c, theta, sigma=sigma, c=np.sqrt(.5))
            i += 1

    return responses
