import numpy as np


def mexican_hat_kernel(x, y, sigma=1.0, k=2.0, alpha=1.0):
    """ Mexican hat kernel (approximated as a difference of Gaussians). More efficient implementation.
    
    Args:
        x, y: the distance between these two variables is used as the argument for the kernel
        sigma: the variance of the first Gaussian component
        k: factor that scales the covariances of the second Gaussian as k*sigma
        alpha: overall scaling factor
        
    Returns:
        mexican hat distance between x and y (scalar if x and y have shape[0] = 1, row vector otherwise)
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    var1 = sigma ** 2
    var2 = (k * sigma) ** 2

    d = np.sum((x - y) ** 2, axis=1)

    # exponential terms for a,b = 1; a,b = 2; a=1, b=2
    d1 = np.exp(-d / (4 * var1))
    d2 = np.exp(-d / (4 * var2))
    dcross = np.exp(-d / (2 * (var1 + var2)))

    result = alpha ** 2 * ((d1 / var1 + d2 / var2) / 4 - dcross / (var1 + var2)) / np.pi

    return result


def fixed_k_mexhat(x, y, sigma=1.0, alpha=1.0):
    """ Mexican hat kernel with k fixed at 2.0
    
    Args:
        x, y: the distance between these two variables is used as the argument for the kernel
        sigma: the variance of the first Gaussian component
        alpha: overall scaling factor
        
    Returns:
        mexican hat distance between x and y (scalar if x and y have shape[0] = 1, row vector otherwise)
    """
    return mexican_hat_kernel(x, y, sigma=sigma, k=2.0, alpha=alpha)


def rbf_kernel(x, y, sigma=1.0):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.exp(- np.sum((x - y) ** 2, axis=1) / sigma ** 2)
