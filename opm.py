import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import filters

def make_opm(size, sigma=4., k=2.):
    """ Generate an orientation preference map (to be used as a fake ground truth). 
     
    
    Args:
        size: dimensionality of the OPM is size x size
        sigma: std of the first gaussian filter
        k: k * sigma is the std of the second gaussian filter
        
    Returns:
        m: complex np.array with shape (size, size)
    """
    
    # generate white noise for real and imaginary map
    a = np.random.randn(size, size)
    b = np.random.randn(size, size)
    
    # apply difference of Gaussians filter to both maps
    a = filters.gaussian_filter(a, sigma) - filters.gaussian_filter(a, k * sigma)
    b = filters.gaussian_filter(b, sigma) - filters.gaussian_filter(b, k * sigma)
    
    # combine real and imaginary parts
    m = a + 1j * b
    
    return m

def plot_opm(m, cmap='hsv', title='Preferred orientation'):
    """ Plot an orientation preference map m.
    
    Args:
        m: orientation preference map. If complex, it is treated like a complex OPM. 
            If real, it is treated like the argument of a complex OPM (the angle, assumed to be between -pi and pi).
        cmap: a matplotlib color map
        
    Returns:
        f, ax: figure and axes of the plot
    """
    
    if np.iscomplex(m).any():
        # compute the preferred orientation (argument)
        # and scale -> theta [0, 180]
        theta = 0.5 * (np.angle(m) + np.pi)
    else:
        theta = m
    
    f, ax = plt.subplots()

    # plot data and adjust axes
    im = ax.imshow(theta, cmap=cmap)
    im.set_clim(0, np.pi)
    loc = np.linspace(0, np.pi, 5) 
    
    # label axes
    labels = ['0', r'$\pi / 4$', r'$\pi / 2$', r'$3 \pi / 4$', r'$\pi$']
    cb = f.colorbar(im, ax=ax)
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title)

    return f, ax

def plot_amplitude_map(m, cmap='jet', title='Amplitude'):
    """ Plot the amplitude of an orientation preference map m.
    
    Args:
        m: orientation preference map. If complex, it is treated like a complex OPM. 
            If real, it is treated like the absolute value of a complex OPM (the angle, assumed to be between -pi and pi).
        cmap: a matplotlib color map
        
    Returns:
        f, ax: figure and axes of the plot
    
    """
    
    if np.iscomplex(m).any():
        # compute the modulus of the orientation map
        A = np.abs(m)
    else:
        A = m
        
    f, ax = plt.subplots()

    im = ax.imshow(A, cmap=cmap)
    

    cb = f.colorbar(im, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title)

    return f, ax

def get_indices(size):
    
    