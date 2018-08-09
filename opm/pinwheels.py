from scipy.interpolate import interp2d
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt


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


# my naive implementation, still not really smooth
def naive_find_pinwheels(m, res=100):
    # size of the map
    sizex, sizey = m.shape

    # interpolate the 2d map (real and imaginary part)
    x, y = np.arange(sizex), np.arange(sizey)
    ai = interp2d(x, y, np.real(m), kind='linear')
    bi = interp2d(x, y, np.imag(m), kind='linear')

    # points at higher resolution to evaluate the interpolated function at
    xx, yy = np.linspace(0, m.shape[0], sizex * res), np.linspace(0, m.shape[1], sizey * res)

    # find zero crossings in real and imaginary part
    zca = np.diff(np.sign(ai(xx, yy))) != 0
    zcb = np.diff(np.sign(bi(xx, yy))) != 0

    zc = np.logical_or(zca, zcb)

    # "indices" of zero crossings
    zcy, zcx = np.where(zc)

    return zcx / res, zcy / res
