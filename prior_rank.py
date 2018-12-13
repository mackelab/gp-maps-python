import numpy as np
import matplotlib.pyplot as plt
import argparse

from opm.gp.kernels import mexican_hat_kernel
from opm.gp.prior import prior_covariance
from opm.gp.helpers import get_2d_indices


def calc_actual_rank(size, atol=1e-7, sigma=6.0, alpha=2.0):
    idx = get_2d_indices(size)
    K = prior_covariance(idx, kernel=mexican_hat_kernel, sigma=sigma, alpha=alpha)

    eigvals = np.linalg.eigvalsh(K)

    return np.where(np.isclose(eigvals[::-1], 0, atol=atol))[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', nargs='+', type=int)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()
    
    size = tuple(args.size)
    
    print("Computing rank of prior covariance matrix...")
    rank = calc_actual_rank(size=size, sigma=args.sigma, alpha=args.alpha)
    print("Rank(K) = {}".format(rank))