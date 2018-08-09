import numpy as np


class FixedNoise:

    def __init__(self, sigma):
        """ Initialize D as variance times identity and G as zeros

        Args:
            sigma: noise covariance matrix
        """

        self.D = sigma

        self.G = np.zeros_like(self.D)


class LowRankNoise:

    def __init__(self, method, q):
        self.method = method
        self.q = q

        self.D = None
        self.G = None

    def fit(self):
        pass
