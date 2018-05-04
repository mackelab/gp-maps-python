import numpy as np

def response(m, cs, theta, c=1.0, sigma=0.1):
    # r(x,s) = a(x) * s_1(x) + b(x) * s_2(x) + c + noise
    return np.real(m) * cs * np.cos(2 * theta) + np.imag(m) * cs * np.sin(2 * theta) + c + np.random.randn(*m.shape) * sigma

def create_stimuli(contrasts, orientations, repetitions):
    
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
                S[i, 2] = 1

                i += 1

    
    return S


def compute_responses(m, contrasts, orientations, repetitions):
    
    responses = []
    
    # for every combination
    for theta in orientations:
        for c in contrasts:
            for j in range(repetitions):
            
                r = response(m, c, theta, sigma=0.1)
                responses.append(r.reshape(-1))

    
    R = np.stack(responses, axis=1).T
    
    return R