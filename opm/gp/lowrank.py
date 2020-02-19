import numpy as np


def lowrank_leftdiv(x, D, G, invR=None, H=None, blocksize=1000):
    """ Calculate (D + G R H')^{-1} x

    Args:
        x: matrix to be divided
        D: n x n diagonal matrix
        G: n x q matrix
        invR: q x q matrix
        H: n x q matrix

    Returns:
        (D + G R H')^{-1} x
    """

    _, N = x.shape
    n, q = G.shape

    if H is None:
        H = G

    if invR is None:
        invR = np.eye(q)

    if x.shape[1] <= blocksize:
        y = np.linalg.solve(D, x)
        y = G.T @ y


    else:
        y = np.zeros(x.shape)
        y2 = np.zeros((q, N))
        numblocks = int(np.ceil(x.shape[1] / blocksize))
        for k in range(1, numblocks + 1):
            index = np.arange((k - 1) * blocksize + 1, np.minimum(k * blocksize, x.shape[1]) + 1)
            index -= 1
            y[:, index] = np.linalg.solve(D, x[:, index])
            y2[:, index] = H.T @ y[:, index]
        y = y2

    if G.shape[1] <= blocksize:
        innerblock = np.linalg.solve(D, G)
        innerblock = invR + H.T @ innerblock


    else:
        numblocks = int(np.ceil(G.shape[1] / blocksize))
        innerblock = np.zeros(G.shape)
        for k in range(1, numblocks + 1):
            index = np.arange((k - 1) * blocksize + 1, np.minimum(k * blocksize, G.shape[1]) + 1)
            index -= 1
            innerblock[:, index] = np.linalg.solve(D, G[:, index])

        innerblock = invR + H.T @ innerblock

    if x.shape[1] <= blocksize:
        y = np.linalg.solve(innerblock, y)
        y = G @ y
        y = np.linalg.solve(D, x - y)

    else:
        numblocks = int(np.ceil(x.shape[1] / blocksize))
        y2 = np.zeros(x.shape)
        for k in range(1, numblocks + 1):
            index = np.arange((k - 1) * blocksize + 1, np.minimum(k * blocksize, x.shape[1]) + 1)
            index -= 1
            y2[:, index] = G @ np.linalg.solve(innerblock, y[:, index])
            y2[:, index] = np.linalg.solve(D, (x[:, index] - y2[:, index]))
        y = y2

    return y


def lowrank_leftdiv_double(x, D, G1, invR1, H1, G2, invR2, H2):
    """ Calculate (D + G1 R1 H1' + G2 R2 H2') ^{-1} x

    Args:
        x: matrix to be divided
        D: n x n diagonal matrix
        G1: n x q1 matrix
        invR1: q1 x q1 matrix
        H1: n x q1 matrix
        G2: n x q2 matrix
        invR2: q2 x q2 matrix
        H2: n x q2 matrix

    Returns:
        (D + G1 R1 H1' + G2 R2 H2') ^{-1} x
    """
    if H1 is None:
        H1 = G1

    if H2 is None:
        H2 = G2

    if invR1 is None:
        invR1 = np.eye(G1.shape[1])

    if invR2 is None:
        invR2 = np.eye(G2.shape[1])

    y = lowrank_leftdiv(x, D, G1, invR1, H1)
    y = H2.T @ y
    innerblock = invR2 + (H2.T @ lowrank_leftdiv(G2, D, G1, invR1, H1))
    y = np.linalg.solve(innerblock, y)
    y = G2 @ y
    y = x - y
    y = lowrank_leftdiv(y, D, G1, invR1, H1)

    return y


def lowrank_leftmult(x, D, G, R=None, H=None):
    """ Calculate (D + G R H') x

    Args:
        x: matrix to be multiplied
        D: n x n diagonal matrix
        G: n x q matrix
        R: q x q matrix
        H: n x q matrix

    Returns:
        (D + G R H') x
    """

    if H is None:
        H = G

    if R is None:
        R = np.eye(G.shape[1])

    y = G @ (R @ (H.T @ x))
    y = D @ x + y

    return y


def premult_by_postcov(x, N, prior, noise):
    """ Left-multiply x by the posterior covariance matrix given by the prior and noise model
        without explicitly computing or storing the posterior covariance.

    Args:
        x: empirical estimate of the map
        N: number of trials
        prior: LowRankPrior object
        D_noise: diagonal noise matrix
        G_noise: low-rank elements of noise

    Returns:

    """
    y = lowrank_leftmult(x, prior.D, prior.G, R=None, H=None)

    beta = 2 / N

    n, q = noise.G.shape

    y = lowrank_leftdiv_double(y, D=prior.D + beta * noise.D, G1=noise.G, invR1=1 / beta * np.eye(q), H1=None,
                               G2=prior.G, invR2=None, H2=None)

    y = x - y

    y = lowrank_leftmult(y, prior.D, prior.G, R=None, H=None)
    return y


def calc_postmean(mhat, N, prior, noise):
    mhat2 = lowrank_leftdiv(mhat, noise.D, noise.G)
    return premult_by_postcov(mhat2, N, prior, noise)
