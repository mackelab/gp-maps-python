import numpy as np

def mexican_hat_kernel(x, y, sigma, k=2.0, alpha=1.0):
    """ Mexican hat kernel (approximated as a difference of Gaussians). More efficient implementation.
    
    Args:
        x, y: the distance between these two variables is used as the argument for the kernel
        sigma: the variance of the first Gaussian component
        k: factor that scales the covariances of the second Gaussian as k*sigma
        alpha: overall scaling factor
        
    Returns:
        mexican hat distance between x and y (scalar if x and y have shape[0] = 1, row vector otherwise)
    """
    var1 = sigma**2
    var2 = (k * sigma)**2
    d = np.sum((x - y)**2, axis=1)
    
    # exponential terms for a,b = 1; a,b = 2; a=1, b=2
    d1 = np.exp(-d / (4 * var1))
    d2 = np.exp(-d / (4 * var2))
    dcross = np.exp(-d / (2 * (var1 + var2)))
    return alpha**2 * ((d1 / var1 + d2 / var2) / 4 - dcross / (var1 + var2)) / np.pi

def dog_kernel(x, y, sigma, k=2.0, alpha=1.0):
    """ Mexican hat kernel (approximated as a difference of Gaussians). More readable implementation
    
    Args:
        x, y: the distance between these two variables is used as the argument for the kernel
        sigma: the variance of the first Gaussian component
        k: factor that scales the covariances of the second Gaussian as k*sigma
        alpha: overall scaling factor
        
    Returns:
        mexican hat distance between x and y (scalar if x and y have shape[0] = 1, row vector otherwise)   
    """
    var = [sigma**2, (k*sigma)**2]
    alpha = [alpha, -alpha]
    
    res = 0
    for a in range(2):
        for b in range(2):
            res += alpha[a] * alpha[b] / (2 * np.pi * (var[a] + var[b])) * np.exp(- np.sum((x-y)**2, axis=1) / (2*(var[a] + var[b])))
    return res
