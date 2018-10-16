import numpy as np


def ridge_cholesky(A, maxtries=5):
    """ Compute a cholesky decomposition of the matrix a, adding a ridge sufficient to make A positive definite
    
    Args:
        A: the matrix to be decomposed
        maxtries: the ridge starts as 1e-6 and icreases up to 1e(-6 + maxtries)
    
    Return:
        L (A = LL') if successful else the current ridge element
    """
    jitter = np.diag(A).mean() * 1e-6

    num_tries = 1
    while num_tries <= maxtries and np.isfinite(jitter):
        try:
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
            return L
        except:
            jitter *= 10
        finally:
            num_tries += 1
    raise np.linalg.LinAlgError("Not positive definite, even with jitter.")

    return jitter


class ICD:
    """Incomplete Cholesky Decomposition: low rank approximation of a kernel matrix.
    """

    def __init__(self, rank, tol=1e-10):
        """
        Args:
        
            rank: Maximal rank of approximation G
            tol: Approximation tolerance
        """
        self.rank = rank
        self.tol = tol
        self.G = None
        self.trained = False

    def fit(self, x, kernel, ridge=1e-4, **kernel_kwargs):
        """Learn the low-rank approximation given a set of indices and a kernel function.
        Args:
            x: The indices at which to compute the covariance (npixels x 2 matrix, where the kth column contains the x and y coordinates of the kth pixel)
            kernel: a kernel function that takes two vectors x and y
            kernel_kwargs: parameters for the kernel function
            
        Returns:
            G (n x self.rank)
        """

        n = x.shape[0]

        # precompute diagonal elements
        D = np.zeros(x.shape[0])
        for k in range(n):
            D[k] = kernel(x[k], x[k], **kernel_kwargs)

        # add ridge to diagonal (for numerical stability)
        D += np.mean(D) * ridge

        # initialize result matrix
        G = np.zeros((n, self.rank))

        # list of remaining columns
        J = set(range(n))
        for i in range(self.rank):

            # find best new element
            jstar = np.argmax(D)
            J.remove(jstar)
            j = list(J)

            # set current diagonal element
            G[jstar, i] = np.sqrt(D[jstar])

            # calculate the i'th column
            newcol = np.zeros(len(j))
            for l in range(len(j)):
                newcol[l] = kernel(x[j[l]], x[jstar], **kernel_kwargs)

            # update the i'th column
            G[j, i] = 1.0 / G[jstar, i] * (newcol - G[j, :] @ G[jstar, :].T)

            # update the diagonal elements
            D[j] = D[j] - (G[j, i] ** 2).ravel()

            # eliminate selected pivot
            D[jstar] = 0

            # check tolerance and rank
            if np.sum(D) < self.tol or i + 1 == self.rank:
                break

        self.G = G[:, :i + 1]
        self.trained = True

        return self.G

    def __call__(self, i, j):
        """ Access portions of K = GG' at indices i, j.
        Args:
            i, j: indices to access K = GG'
        Return:
            K[i,j]
        """
        if not self.trained:
            raise RuntimeError('Call train(...) first!')
        return self.G[i, :] @ self.G[j, :].T

    def __getitem__(self, idx):
        """ Access portions of K = GG'
        Args:
            item: indices to access K = GG'
        Return:
            K[idx[0], idx[i]]
        """

        return self(idx[0], idx[1])
