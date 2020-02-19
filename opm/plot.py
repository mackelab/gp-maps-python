import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from skimage.measure import find_contours


def rgb_from_opm(m, cmap="hsv", shade=False, rmin=10, rmax=80):
    if np.iscomplex(m).any():
        # compute the preferred orientation (argument)
        # and scale -> theta [0, 180]
        theta = 0.5 * np.angle(m) % np.pi
    else:
        theta = m

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

    theta_rgb[:, :, 3][np.isnan(theta)] = 0

    return theta, theta_rgb


def plot_opm(m, cmap='hsv', title='Preferred orientation', pinwheels=False, shade=False, rmin=10, rmax=80, ax=None,
             colorbar=True):
    """ Plot an orientation preference map m.
    
    Args:
        m: orientation preference map. If complex, it is treated like a complex OPM. 
            If real, it is treated like the argument of a complex OPM (the angle, assumed to be between -pi and pi).
        cmap: a matplotlib color map
        
    Returns:
        f, ax: figure and axes of the plot
    """

    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()

    theta, theta_rgb = rgb_from_opm(m=m, cmap=cmap, shade=shade, rmin=rmin, rmax=rmax)

    # plot data and adjust axes
    im = ax.imshow(theta, cmap=cmap)
    im_rgb = ax.imshow(theta_rgb)
    im.set_clim(0, np.pi)
    loc = np.linspace(0, np.pi, 5)

    # add colorbar
    if colorbar:
        labels = ['0', r'$\pi / 4$', r'$\pi / 2$', r'$3 \pi / 4$', r'$\pi$']
        cb = f.colorbar(im, ax=ax)
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)

    # remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title)

    if pinwheels:
        if not np.iscomplex(m).any():
            raise ValueError('Map must be complex in order to compute pinwheels')
        else:
            plot_pinwheels(m, ax)

    return f, ax, im_rgb


def plot_amplitude_map(m, cmap='jet', title='Amplitude', ax=None, colorbar=False):
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

    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()

    im = ax.imshow(A, cmap=cmap)

    if colorbar:
        cb = f.colorbar(im, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
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


def plot_orientation_histogram(m, weighted=False, bins=20, ax=None, polar=False, **kwargs):
    if not ax:
        if polar:
            f, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            f, ax = plt.subplots()
    else:
        f = None

    weights = np.abs(m).reshape(-1) if weighted else None
    theta = (np.angle(m).reshape(-1)) * 0.5 % np.pi

    if polar:
        weights = np.append(weights, weights) if weighted else None
        r, theta, patches = ax.hist(np.append(theta, theta + np.pi), range=(0, 2 * np.pi), weights=weights,
                                    bins=bins, **kwargs)
    else:
        ax.hist(np.rad2deg(theta), bins=bins, weights=weights, **kwargs)
        ax.set_xticks([0, 45, 90, 135, 180])

    return f, ax


def polar_histogram(x, ax=None, **kwargs):
    """ Plot a histogram on polar coordinates (radius is relative frequency)
    
    Args:
        x: np array containing the data
        ax: pyplot axis to be plotted on. if not given, a new axis is created via subplots
        kwargs: arguments passed to https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html, 
                after setting default color
    Returns:
        r: frequency / density
        theta: bin edges
        patches: matplotlib patches objects
    """

    # set default arguments for histogram
    kwargs = dict({'color': 'white', 'edgecolor': 'C0'}, **kwargs)

    # if no ax is given
    if not ax:
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # do the plotting
    r, theta, patches = ax.hist(np.append(x, x + np.pi), range=(0, 2 * np.pi), **kwargs)

    return r, theta, patches


def animate(trial):
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig,
                                    [[plt.imshow(frame, vmin=trial.min(), vmax=trial.max(), animated=True)] for
                                     frame in
                                     trial], interval=200, repeat_delay=1000)

    return fig, ani
