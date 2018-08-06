import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import filters

from .pinwheels import plot_pinwheels

def make_opm(size, sigma=4., k=2., alpha=1.):
    """ Generate an orientation preference map (to be used as a fake ground truth). 
     
    
    Args:
        size: dimensionality of the OPM is size x size
        sigma: std of the first gaussian filter
        k: k * sigma is the std of the second gaussian filter
        
    Returns:
        m: complex np.array with shape (size, size)
    """
    
    if isinstance(size, int):
        sx, sy = size, size
    else:
        sx, sy = size
    
    # generate white noise for real and imaginary map
    a = np.random.randn(sx, sy)
    b = np.random.randn(sx, sy)
    
    # apply difference of Gaussians filter to both maps
    a = alpha * filters.gaussian_filter(a, sigma) - alpha * filters.gaussian_filter(a, k * sigma)
    b = alpha * filters.gaussian_filter(b, sigma) - alpha * filters.gaussian_filter(b, k * sigma)
    
    # combine real and imaginary parts
    m = a + 1j * b
    
    return m

def plot_opm(m, cmap='hsv', title='Preferred orientation', pinwheels=True, shade=False, rmin=10, rmax=80):
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

    r = np.abs(m)

    cmap = cm.get_cmap('hsv', 128)

    theta_rgb = cmap(theta / np.pi)

    if shade:
        rmin = np.percentile(r, rmin)
        rmax = np.percentile(r, rmax)
        r = np.minimum(r, rmax)
        r = np.maximum(r, rmin)
        r = (r - rmin) / (rmax - rmin)
        for i in range(3):
            theta_rgb[:,:,i] = theta_rgb[:,:,i] + (1 - theta_rgb[:,:,i]) * (1 - r)


    # plot data and adjust axes
    im = ax.imshow(theta, cmap=cmap)
    ax.imshow(theta_rgb)
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
    
    if pinwheels:
        if not np.iscomplex(m).any():
            raise ValueError('Map must be complex in order to compute pinwheels')
        else:
            plot_pinwheels(m, ax)

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




def calculate_map(responses, stimuli):
    """ Compute OPM components from an experiment (max likelihood solution)
    
    Args:
        stimuli: N_cond x N_rep x d array, stimulus conditions for each trial
        responses: N_cond x N_rep x n_x x n_y array, responses from an experiment 
        
    Returns: estimated map components: d x n_x x n_y array 
    """
    V = stimuli
    d = V.shape[2]
    R = responses
    N_c, N_r, nx, ny = R.shape
    N = N_c * N_r
    n = int(R.size / N)
    
    V = V.reshape((N,d))
    
    # least squares estimate of real and imaginary components of the map
    M_flat = np.linalg.inv(V.T @ V) @ V.T @ R.reshape((N,n))

        
    # reshape map into three
    M = np.zeros((d, nx, ny))
    for i, a in enumerate(M_flat):
        M[i] = a.reshape(nx, ny)
        
    return M
    