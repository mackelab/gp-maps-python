import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import find_contours


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

    colormap = cm.get_cmap(cmap, 128)

    theta_rgb = colormap(theta / np.pi)

    if shade:
        rmin = np.percentile(r, rmin)
        rmax = np.percentile(r, rmax)
        r = np.minimum(r, rmax)
        r = np.maximum(r, rmin)
        r = (r - rmin) / (rmax - rmin)
        for i in range(3):
            theta_rgb[:, :, i] = theta_rgb[:, :, i] + (1 - theta_rgb[:, :, i]) * (1 - r)

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


def plot_pinwheels(m, ax=None, color='white', linewidth=2):
    """ Plot the pinwheels (zero crossings) of an orientation preference map
    
    Args:
        m: the orientation preference map (complex)
        ax: matplotlib axis to plot on. If not given, new figure is initialized
        color: line color for plt.plot(...)
        linewidth: argument for plt.plot(...)
        
    Returns:
        list of pinwheel contours (consists of 2d points)
    """

    # if an ax is given as an argument, we don't need to initialize a new one
    if not ax:
        f, ax = plt.subplots()

    # use skimage's function for contour lines to find zero crossings
    c = find_contours(np.real(m), 0) + find_contours(np.imag(m), 0)

    # plot all the contours
    for contour in c:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)

    return c
    