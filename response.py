import numpy as np

def response(m, cs, theta, c=1.0, sigma=0.1):
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
    return np.real(m) * cs * np.cos(2 * theta) + np.imag(m) * cs * np.sin(2 * theta) + c + np.random.randn(*m.shape) * sigma

def create_stimuli(contrasts, orientations, repetitions):
    """ Compute stimulus condition matrix
    
    Args:
        contrasts: list of contrast conditions
        orientations: list of orientation conditions (radians)
        repetitions: number of repetitions (integer)
        
    Returns:
        ntrials x 3 matrix containing [c * cos(2theta), c * sin(2theta), 1] for each trial
    """
    
    # initialize size and array
    N = len(contrasts) * len(orientations)
    S = np.zeros((N * repetitions, 3))
    
    i = 0
    
    # for every combination
    for theta in orientations:
        for c in contrasts:
            for j in range(repetitions):

                S[i, 0] = c * np.cos(2 * theta)
                S[i, 1] = c * np.sin(2 * theta)
                S[i, 2] = np.sqrt(.5)

                i += 1

    
    return S


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
    
    responses = []
    
    # for every combination
    for theta in orientations:
        for c in contrasts:
            for j in range(repetitions):
            
                r = response(m, c, theta, sigma=sigma)
                responses.append(r.reshape(-1))

    
    R = np.stack(responses, axis=1).T
    
    return R