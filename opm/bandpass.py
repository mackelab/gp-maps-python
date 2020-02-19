from scipy.ndimage import gaussian_filter


def spatial_bandpass(img, high=960, low=168, pixelsize=8, binning=2):
    """ Perform spatial bandpass filtering with Gaussian filters on a set of images.
        Only the last two dimensions are treated as image dimensions! The other dimensions are not filtered across.

    Args:
        img (np.array): np.array of arbitrary shape, whose last two dimensions are image dimensions
        high (float): FWHM of the high-pass Gaussian in micrometer
        low (float): FWHM of the low-pass Gaussian in micrometer
        pixelsize (float): pixelsize in micrometer
        binning (int): how many pixels were binned prior to the filtering

    Returns:
        np.array of the same shape, bandpass filtered
    """

    fwhm_h = high / (pixelsize * binning)
    fwhm_l = low / (pixelsize * binning)

    s = 1 / 2.3548  # FWHM = 2.3548 * sigma
    sigma_h = fwhm_h * s
    sigma_l = fwhm_l * s

    # number of non-image dimensions
    dim = len(img.shape) - 2

    sigmas_h = [0.] * dim + [sigma_h] * 2
    sigmas_l = [0.] * dim + [sigma_l] * 2

    img_bp = gaussian_filter(img, sigma=sigmas_l, mode="nearest") - gaussian_filter(img, sigma=sigmas_h, mode="nearest")

    return img_bp
