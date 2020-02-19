import numpy as np


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
    S = np.zeros((N, repetitions, 3))

    i = 0

    # for every combination
    for theta in orientations:
        for c in contrasts:
            for j in range(repetitions):
                S[i, j, 0] = c * np.cos(2 * theta)
                S[i, j, 1] = c * np.sin(2 * theta)
                S[i, j, 2] = np.sqrt(.5)

            i += 1

    return S
