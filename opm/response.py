import numpy as np




def compute_responses(m, stim, noise=0.01):
    """ Compute matrix of responses
    
    Args:
        m: (d, nx, ny) orientation preference map
        stim: (N_cond, N_rep, d) stimuli
        
    Returns: 
        ntrials x npixels matrix containing responses at each pixel for each trial
    """

    d, nx, ny = m.shape
    n = nx * ny
    m = m.reshape(d, n)

    N = stim.shape[0] * stim.shape[1]
    d = stim.shape[2]

    V = stim.reshape(N, d)

    R = V @ m
    R = R + np.random.randn(*R.shape) * noise

    R = R.reshape(stim.shape[0], stim.shape[1], nx, ny)

    return R
